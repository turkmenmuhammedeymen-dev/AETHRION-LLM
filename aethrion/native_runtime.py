"""Verified local-only inference runtime for released AETHRION model artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
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
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024.")
        model = self.model(capability)
        tokenizer = self.tokenizer(capability)
        context = self._text_context(tokenizer, model, prompt, self.device)
        result = model.generate(
            context, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        output = tokenizer.decode(result[0, context.size(1):].tolist())
        return {"text": output, "model": f"AETHRION-{capability.upper()}"}

    @torch.no_grad()
    def generate_image(
        self, prompt: str, output_path: str | Path | None = None,
        *, inference_steps: int = 50
    ) -> Path:
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
        model = self.model("vision")
        tokenizer = self.tokenizer("vision")
        config = model.config
        with Image.open(image_path) as source:
            source = source.convert("RGB").resize((config.image_size, config.image_size))
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
