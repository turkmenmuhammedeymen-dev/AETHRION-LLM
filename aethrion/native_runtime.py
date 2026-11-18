"""Verified local-only inference runtime for released AETHRION model artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aethrion.native_registry import NativeModelRegistry, NativeModelUnavailable


class NativeRuntimeError(RuntimeError):
    """Native inference failed due to missing or invalid local artifacts."""


class NativeModelRuntime:
    """Load only registry-approved local checkpoints; never resolve network URLs."""

    def __init__(
        self,
        root: str | Path = ".",
        manifest: str | Path = "models/manifest.json",
        device: str | torch.device | None = None,
    ):
        self.root = Path(root).resolve()
        self.registry = NativeModelRegistry(self.root, manifest)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._models: dict[str, nn.Module] = {}
        self._tokenizers: dict[str, Any] = {}
        self._processors: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def status(self) -> dict[str, str]:
        return {name: self.registry.status(name) for name in self.registry.capabilities()}

    def _build_model(self, capability: str, config: dict[str, Any]) -> nn.Module:
        if capability == "chat":
            from aethrion.model import AETHRIONLM, ModelConfig
            return AETHRIONLM(ModelConfig(**config))
        if capability == "code":
            from aethrion.code_model import AETHRIONCodeLM, CodeModelConfig
            return AETHRIONCodeLM(CodeModelConfig(**config))
        if capability == "vision":
            from aethrion.vision_model import AETHRIONVision, VisionConfig
            return AETHRIONVision(VisionConfig(**config))
        if capability == "image":
            from aethrion.image_model import AETHRIONImageGenerator, ImageModelConfig
            return AETHRIONImageGenerator(ImageModelConfig(**config))
        if capability == "asr":
            from aethrion.asr_model import AETHRIONASR, ASRConfig
            return AETHRIONASR(ASRConfig(**config))
        if capability == "tts":
            from aethrion.tts_model import AETHRIONTTS, TTSConfig
            return AETHRIONTTS(TTSConfig(**config))
        if capability == "vocoder":
            from aethrion.tts_model import AETHRIONVocoder
            return AETHRIONVocoder(**config)
        if capability == "3d":
            from aethrion.model_3d import AETHRION3D, Model3DConfig
            return AETHRION3D(Model3DConfig(**config))
        if capability == "research":
            from aethrion.research_model import AETHRIONResearch, ResearchConfig
            return AETHRIONResearch(ResearchConfig(**config))
        if capability == "document":
            from aethrion.document_model import AETHRIONDocument, DocumentConfig
            return AETHRIONDocument(DocumentConfig(**config))
        if capability == "safety":
            from aethrion.safety_model import AETHRIONSafe, SafetyConfig
            return AETHRIONSafe(SafetyConfig(**config))
        raise NativeRuntimeError(f"Unsupported native capability: {capability}")

    def model(self, capability: str) -> nn.Module:
        if capability in self._models:
            return self._models[capability]
        try:
            artifacts = self.registry.resolve(capability)
        except NativeModelUnavailable as exc:
            raise NativeModelUnavailable(str(exc)) from exc
        model = self._build_model(capability, artifacts.model_config)
        try:
            if capability == "chat":
                from aethrion.checkpoint import load_checkpoint
                state = load_checkpoint(artifacts.checkpoint, map_location="cpu")
                if state.get("model_config") != artifacts.model_config:
                    raise NativeRuntimeError("BASE checkpoint model_config differs from its signed manifest.")
                weights = state.get("model")
            else:
                state = torch.load(
                    artifacts.checkpoint, map_location="cpu", weights_only=True
                )
                if not isinstance(state, dict) or state.get("format") != "aethrion-specialist-checkpoint-v1":
                    raise NativeRuntimeError(f"Unsupported {capability} checkpoint format.")
                if state.get("model_id") != capability:
                    raise NativeRuntimeError(f"Checkpoint model_id mismatch for {capability}.")
                if state.get("model_config") != artifacts.model_config:
                    raise NativeRuntimeError(f"Checkpoint model_config mismatch for {capability}.")
                training_report = json.loads(
                    artifacts.training_report.read_text(encoding="utf-8")
                )
                if state.get("dataset_sha256") != training_report.get("dataset_sha256"):
                    raise NativeRuntimeError(f"Checkpoint dataset hash mismatch for {capability}.")
                weights = state.get("model_state")
            if not isinstance(weights, dict):
                raise NativeRuntimeError(f"Checkpoint has no model weights for {capability}.")
            model.load_state_dict(weights, strict=True)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            if isinstance(exc, NativeRuntimeError):
                raise
            raise NativeRuntimeError(f"Could not load local {capability} checkpoint: {type(exc).__name__}") from exc
        model.to(self.device)
        model.eval()
        self._models[capability] = model
        return model

    def tokenizer(self, capability: str):
        if capability in self._tokenizers:
            return self._tokenizers[capability]
        artifacts = self.registry.resolve(capability)
        if artifacts.tokenizer is None:
            raise NativeRuntimeError(f"AETHRION-{capability.upper()} has no registered tokenizer.")
        from aethrion.tokenizer import ByteBPETokenizer
        tokenizer = ByteBPETokenizer.load(artifacts.tokenizer)
        expected_vocab = artifacts.model_config.get("vocab_size")
        if expected_vocab is not None and tokenizer.vocab_size != int(expected_vocab):
            raise NativeRuntimeError(
                f"Tokenizer vocabulary {tokenizer.vocab_size} does not match {capability} model vocabulary {expected_vocab}."
            )
        self._tokenizers[capability] = tokenizer
        return tokenizer

    def processor(self, capability: str) -> dict[str, Any]:
        if capability in self._processors:
            return self._processors[capability]
        artifacts = self.registry.resolve(capability)
        if artifacts.processor is None:
            raise NativeRuntimeError(f"AETHRION-{capability.upper()} has no registered processor metadata.")
        try:
            payload = json.loads(artifacts.processor.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NativeRuntimeError(f"Invalid processor metadata for {capability}.") from exc
        if not isinstance(payload, dict):
            raise NativeRuntimeError(f"Processor metadata for {capability} must be a JSON object.")
        self._processors[capability] = payload
        return payload

    @staticmethod
    def _text_context(tokenizer, model, text: str, device: torch.device):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must not be empty.")
        ids = tokenizer.encode(text)
        if not ids:
            raise ValueError("Input text produced no tokenizer IDs.")
        config = getattr(model, "config", None)
        limit = getattr(config, "block_size", None)
        if limit is None:
            limit = getattr(config, "max_text_length", None)
        if limit is None:
            limit = getattr(config, "text_length", None)
        if limit is None:
            limit = getattr(config, "max_length", 512)
        ids = ids[-int(limit):]
        return torch.tensor([ids], dtype=torch.long, device=device)

    @torch.no_grad()
    def generate_text(
        self,
        capability: str,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        if capability not in {"chat", "code"}:
            raise ValueError("Text generation is available only for BASE or CODE.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Input text must not be empty.")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer.")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not np.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise ValueError("temperature must be a finite number greater than zero.")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be None or a positive integer.")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not np.isfinite(float(top_p))
            or not 0.0 < float(top_p) <= 1.0
        ):
            raise ValueError("top_p must be a finite number in (0, 1].")
        model = self.model(capability)
        tokenizer = self.tokenizer(capability)
        context = self._text_context(tokenizer, model, prompt, self.device)
        result = model.generate(
            context, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        output = tokenizer.decode(result[0, context.size(1):].tolist())
        return {"text": output, "model": f"AETHRION-{capability.upper()}"}

    @staticmethod
    def _validate_inference_steps(inference_steps: int) -> None:
        if (
            isinstance(inference_steps, bool)
            or not isinstance(inference_steps, int)
            or not 2 <= inference_steps <= 100
        ):
            raise ValueError("inference_steps must be an integer between 2 and 100.")

    @torch.no_grad()
    def generate_image(
        self, prompt: str, output_path: str | Path | None = None,
        *, inference_steps: int = 50
    ) -> Path:
        self._validate_inference_steps(inference_steps)
        model = self.model("image")
        tokenizer = self.tokenizer("image")
        context = self._text_context(tokenizer, model, prompt, self.device)
        image = model.generate(context, inference_steps=inference_steps)[0]
        array = (
            (image.detach().cpu().permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5
        ).round().to(torch.uint8).numpy()
        try:
            from PIL import Image
        except ImportError as exc:
            raise NativeRuntimeError("Local image export requires Pillow; no hosted image service is used.") from exc
        destination = Path(output_path) if output_path else Path(tempfile.gettempdir()) / (
            "aethrion-image-" + hashlib.sha256(array.tobytes()).hexdigest()[:12] + ".png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(destination, format="PNG")
        return destination

    @torch.no_grad()
    def image_text_similarity(self, image_path: str | Path, text: str) -> float:
        from PIL import Image
        image_file = Path(image_path)
        if (
            not image_file.is_file() or image_file.stat().st_size < 1
            or image_file.stat().st_size > 20 * 1024 * 1024
        ):
            raise NativeRuntimeError("Image must be a local file between 1 byte and 20 MiB.")
        with Image.open(image_file) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise NativeRuntimeError("Only PNG, JPEG and WebP images are supported.")
            width, height = source.size
            if (
                width < 1 or height < 1 or width > 8192 or height > 8192
                or width * height > 40_000_000
            ):
                raise NativeRuntimeError(
                    "Image exceeds safe dimensions (8192 per side; 40 megapixels total)."
                )
            source.load()
            model = self.model("vision")
            tokenizer = self.tokenizer("vision")
            target_size = int(model.config.image_size)
            if not 16 <= target_size <= 1024:
                raise NativeRuntimeError("Registered vision image_size is outside safe inference bounds.")
            source = source.convert("RGB").resize((target_size, target_size))
            array = np.asarray(source, dtype=np.float32) / 127.5 - 1.0
        image = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        context = self._text_context(tokenizer, model, text, self.device)
        visual = model.encode_image(image)
        textual = model.encode_text(context)
        return float((visual * textual).sum(dim=-1).item())

    @torch.no_grad()
    def transcribe_wav(self, audio_path: str | Path) -> str:
        from aethrion.audio_features import read_wav, log_mel_spectrogram
        model = self.model("asr")
        tokenizer = self.tokenizer("asr")
        waveform, sample_rate = read_wav(audio_path, target_sample_rate=16000)
        features = log_mel_spectrogram(
            waveform, sample_rate, n_mels=model.config.n_mels
        ).unsqueeze(0).to(self.device)
        input_lengths = torch.tensor([features.size(1)], device=self.device)
        log_probs, output_lengths = model(features, input_lengths)
        ids = log_probs[:int(output_lengths[0].item()), 0].argmax(dim=-1).tolist()
        collapsed = []
        previous = None
        blank = model.config.blank_id
        for token_id in ids:
            if token_id != previous and token_id != blank:
                collapsed.append(token_id)
            previous = token_id
        return tokenizer.decode(collapsed).strip()

    @torch.no_grad()
    def synthesize_wav(self, text: str, output_path: str | Path | None = None) -> Path:
        from aethrion.audio_features import write_wav
        acoustic = self.model("tts")
        vocoder = self.model("vocoder")
        tokenizer = self.tokenizer("tts")
        context = self._text_context(tokenizer, acoustic, text, self.device)
        durations = acoustic.predict_durations(context, min_frames=1, max_frames=200)
        alignment, lengths = acoustic.durations_to_alignment(durations)
        frame_count = int(lengths[0].item())
        mel = acoustic(context, alignment[:, :frame_count])
        waveform = vocoder(mel.transpose(1, 2))[0]
        processor = self.processor("tts")
        sample_rate = int(processor.get("sample_rate", 16000))
        if not 8000 <= sample_rate <= 96000:
            raise NativeRuntimeError("TTS processor sample_rate must be between 8000 and 96000.")
        destination = Path(output_path) if output_path else Path(tempfile.gettempdir()) / (
            "aethrion-speech-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] + ".wav"
        )
        return write_wav(destination, waveform, sample_rate)

    @torch.no_grad()
    def generate_point_cloud(self, prompt: str, output_path: str | Path | None = None,
                             *, inference_steps: int = 50) -> Path:
        self._validate_inference_steps(inference_steps)
        from aethrion.model_3d import point_cloud_to_obj
        model = self.model("3d")
        tokenizer = self.tokenizer("3d")
        context = self._text_context(tokenizer, model, prompt, self.device)
        points = model.generate(context, inference_steps=inference_steps)[0]
        obj_text = point_cloud_to_obj(points)
        destination = Path(output_path) if output_path else Path(tempfile.gettempdir()) / (
            "aethrion-point-cloud-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12] + ".obj"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(obj_text, encoding="utf-8")
        return destination

    def handle(self, capability: str, request: Any) -> dict[str, Any]:
        """Adapt a validated local model call to the shared capability router."""
        prompt = getattr(request, "prompt", "")
        options = getattr(request, "options", {})
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("Request options must be an object.")
        if capability in {"chat", "code"}:
            return self.generate_text(
                capability, prompt,
                max_new_tokens=options.get("max_new_tokens", 256),
                temperature=options.get("temperature", 0.8),
                top_k=options.get("top_k", 50),
                top_p=options.get("top_p", 0.95),
            )
        if capability == "image":
            path = self.generate_image(
                prompt, inference_steps=options.get("inference_steps", 50)
            )
            return {"image_path": str(path), "format": "png", "model": "AETHRION-IMAGE"}
        if capability == "vision":
            image_path = options.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("vision inference requires a local image_path option.")
            score = self.image_text_similarity(image_path, prompt)
            return {"similarity": score, "model": "AETHRION-VISION",
                    "meaning": "image/text embedding similarity, not a natural-language image description"}
        if capability == "asr":
            audio_path = options.get("audio_path")
            if not isinstance(audio_path, str) or not audio_path:
                raise ValueError("ASR inference requires a local audio_path option.")
            return {"text": self.transcribe_wav(audio_path), "model": "AETHRION-ASR"}
        if capability == "tts":
            output = self.synthesize_wav(prompt)
            return {"audio_path": str(output), "format": "wav", "model": "AETHRION-TTS"}
        if capability == "3d":
            output = self.generate_point_cloud(
                prompt,
                inference_steps=options.get("inference_steps", 50)
            )
            return {"asset_path": str(output), "format": "obj-point-cloud",
                    "model": "AETHRION-3D",
                    "note": "Vertex-only point cloud; not a watertight mesh or textured asset."}
        if capability == "research":
            return self.search_research(prompt, top_k=options.get("top_k", 5))
        if capability == "document":
            document_path = options.get("document_path")
            if not isinstance(document_path, str) or not document_path:
                raise ValueError("Document inference requires a local document_path option.")
            return self.analyze_document(document_path)
        if capability == "safety":
            return self.classify_safety(prompt)
        raise NativeRuntimeError(
            f"Local inference adapter for '{capability}' is not yet implemented or released."
        )

    def validate_research_index(self) -> bool:
        """Validate local research index artifacts before registering RESEARCH as ready."""
        processor = self.processor("research")
        if processor.get("format") != "aethrion-local-research-index-v1":
            raise NativeRuntimeError("Research processor does not describe a supported local index.")
        artifacts = self.registry.resolve("research")

        def verified_file(path_key: str, hash_key: str, max_bytes: int) -> Path:
            relative = processor.get(path_key)
            expected = processor.get(hash_key)
            if not isinstance(relative, str) or not relative.strip() or not isinstance(expected, str):
                raise NativeRuntimeError(f"Research processor is missing {path_key}/{hash_key}.")
            candidate = Path(relative)
            if (
                candidate.is_absolute() or chr(92) in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
            ):
                raise NativeRuntimeError("Research index paths must be safe repository-relative paths.")
            unresolved = self.root / candidate
            if unresolved.is_symlink():
                raise NativeRuntimeError("Research index artifacts cannot be symbolic links.")
            resolved = unresolved.resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise NativeRuntimeError("Research index path escapes repository root.") from exc
            if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size < 1:
                raise NativeRuntimeError(f"Research index artifact is missing: {path_key}.")
            if resolved.stat().st_size > max_bytes or self._digest(resolved) != expected.lower():
                raise NativeRuntimeError(f"Research index size or SHA-256 check failed: {path_key}.")
            return resolved

        index_path = verified_file("index_path", "index_sha256", 512 * 1024 * 1024)
        metadata_path = verified_file("metadata_path", "metadata_sha256", 64 * 1024 * 1024)
        if processor.get("tokenizer_sha256") != self._digest(artifacts.tokenizer):
            raise NativeRuntimeError("Research index tokenizer hash does not match the released model tokenizer.")
        from aethrion.tokenizer import ByteBPETokenizer
        tokenizer = ByteBPETokenizer.load(artifacts.tokenizer)
        try:
            with zipfile.ZipFile(index_path) as archive:
                entries = [info for info in archive.infolist() if not info.is_dir()]
                if not entries or len(entries) > 32 or sum(info.file_size for info in entries) > 512 * 1024 * 1024:
                    raise NativeRuntimeError("Research NPZ archive exceeds safe entry or expanded-size limits.")
            with np.load(index_path, allow_pickle=False) as arrays:
                passage_ids = np.array(arrays["passage_ids"], copy=True)
                passage_mask = np.array(arrays["passage_mask"], copy=True)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except NativeRuntimeError:
            raise
        except (OSError, ValueError, KeyError, EOFError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise NativeRuntimeError("Research index artifacts are invalid or unreadable.") from exc
        if (
            passage_ids.ndim != 2 or passage_ids.shape[0] < 1 or passage_ids.shape[1] < 2
            or passage_mask.shape != passage_ids.shape
            or not np.issubdtype(passage_ids.dtype, np.integer)
            or not np.issubdtype(passage_mask.dtype, np.number)
            or not np.isfinite(passage_ids).all()
            or not np.isfinite(passage_mask).all()
            or np.any(passage_ids < 0)
            or np.any(passage_ids >= tokenizer.vocab_size)
            or np.any((passage_mask != 0) & (passage_mask != 1))
        ):
            raise NativeRuntimeError("Research index token or mask arrays have invalid shapes or values.")
        required_metadata = ("title", "source_uri", "license", "evidence_uri", "excerpt", "doc_id")
        if not isinstance(metadata, list) or len(metadata) != passage_ids.shape[0]:
            raise NativeRuntimeError("Research metadata count does not match index rows.")
        for row in metadata:
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in required_metadata
            ):
                raise NativeRuntimeError("Research metadata has missing or invalid evidence fields.")
            if len(row["excerpt"]) > 12000:
                raise NativeRuntimeError("Research metadata excerpt exceeds its text limit.")
        return True

    @torch.no_grad()
    def search_research(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve evidence from a previously built and hash-verified local corpus index."""
        if (
            isinstance(top_k, bool) or not isinstance(top_k, int)
            or not 1 <= top_k <= 20
        ):
            raise ValueError("top_k must be an integer between 1 and 20.")
        model = self.model("research")
        tokenizer = self.tokenizer("research")
        processor = self.processor("research")
        if processor.get("format") != "aethrion-local-research-index-v1":
            raise NativeRuntimeError("Research processor does not describe an AETHRION local index.")
        def verified_local_file(path_key: str, hash_key: str) -> Path:
            relative = processor.get(path_key)
            expected = processor.get(hash_key)
            if not isinstance(relative, str) or not relative or not isinstance(expected, str):
                raise NativeRuntimeError(f"Research processor missing {path_key}/{hash_key}.")
            candidate = Path(relative)
            if candidate.is_absolute() or chr(92) in relative or any(x in {"", ".", ".."} for x in relative.split("/")):
                raise NativeRuntimeError("Research index paths must be safe repository-relative paths.")
            resolved = (self.root / candidate).resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise NativeRuntimeError("Research index path escapes repository root.") from exc
            if not resolved.is_file() or self._digest(resolved) != expected.lower():
                raise NativeRuntimeError(f"Research index integrity failure: {path_key}.")
            return resolved
        index_path = verified_local_file("index_path", "index_sha256")
        metadata_path = verified_local_file("metadata_path", "metadata_sha256")
        if processor.get("tokenizer_sha256") != self._digest(self.registry.resolve("research").tokenizer):
            raise NativeRuntimeError("Research index tokenizer hash does not match the released model tokenizer.")
        try:
            with np.load(index_path, allow_pickle=False) as archive:
                passage_ids = np.array(archive["passage_ids"], copy=True)
                passage_mask = np.array(archive["passage_mask"], copy=True)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise NativeRuntimeError("Could not load verified local research index.") from exc
        if passage_ids.ndim != 2 or passage_mask.shape != passage_ids.shape:
            raise NativeRuntimeError("Research index token and mask arrays have invalid shapes.")
        if not isinstance(metadata, list) or len(metadata) != passage_ids.shape[0]:
            raise NativeRuntimeError("Research metadata count does not match index rows.")
        context = self._text_context(tokenizer, model, query, self.device)
        passage_tensor = torch.as_tensor(passage_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.as_tensor(passage_mask, dtype=torch.bool, device=self.device)
        scores, indices = model.retrieve(context, passage_tensor, mask_tensor, top_k=top_k)
        results = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist()):
            record = metadata[int(index)]
            results.append({
                "title": record["title"],
                "source_uri": record["source_uri"],
                "license": record["license"],
                "evidence_uri": record["evidence_uri"],
                "excerpt": record["excerpt"],
                "score": float(score),
                "doc_id": record["doc_id"],
            })
        return {"results": results, "index_type": "local_only", "model": "AETHRION-RESEARCH"}

    @torch.no_grad()
    def analyze_document(self, document_path: str | Path) -> dict[str, Any]:
        """Run the released text/layout classifier over locally extracted document text.

        The current preprocessing flattens extracted text and assigns a full-page
        placeholder box. It is not OCR and does not preserve true page geometry.
        """
        from aethrion.document_tools import extract_document_text
        model = self.model("document")
        tokenizer = self.tokenizer("document")
        processor = self.processor("document")
        text = extract_document_text(document_path)
        context = self._text_context(tokenizer, model, text, self.device)
        length = context.size(1)
        boxes = torch.zeros((1, length, 4), dtype=torch.float32, device=self.device)
        boxes[..., 2] = 1.0
        boxes[..., 3] = 1.0
        layout_ids = torch.zeros((1, length), dtype=torch.long, device=self.device)
        output = model(context, boxes, layout_ids)
        tag_ids = output["tag_logits"][0].argmax(dim=-1).tolist()
        tag_names = processor.get("tag_labels", [])
        labels = [
            tag_names[tag_id] if isinstance(tag_names, list) and tag_id < len(tag_names)
            else f"TAG_{tag_id}"
            for tag_id in tag_ids
        ]
        start, end = output["qa_logits"][0].argmax(dim=-1).tolist()
        if end < start:
            start, end = end, start
        answer = tokenizer.decode(context[0, start:end + 1].tolist())
        return {
            "text": text,
            "token_tags": labels,
            "extractive_span": answer,
            "model": "AETHRION-DOC",
            "layout_mode": "flattened_text_placeholder",
            "limitations": "No OCR or true page-coordinate preservation is performed by this adapter.",
        }

    @torch.no_grad()
    def classify_safety(self, text: str) -> dict[str, Any]:
        model = self.model("safety")
        tokenizer = self.tokenizer("safety")
        context = self._text_context(tokenizer, model, text, self.device)
        output = model.predict_candidate(context)
        return {
            "decision": output["decision"][0],
            "confidence": float(output["confidence"][0].item()),
            "reason_scores": output["reason_scores"][0].detach().cpu().tolist(),
            "model": "AETHRION-SAFE",
            "note": "Candidate classification only; must be calibrated and independently evaluated before deployment.",
        }
