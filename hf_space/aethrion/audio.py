"""Bounded turn-based speech recognition and synthesis adapter."""
from __future__ import annotations
import os
from pathlib import Path
import httpx

class AudioProviderError(RuntimeError):
    pass

class OpenAISpeechProvider:
    MAX_AUDIO_BYTES = 25 * 1024 * 1024
    MAX_TEXT_CHARS = 4096
    VOICES = {"alloy","ash","ballad","coral","echo","fable","onyx","nova","sage","shimmer","verse","marin","cedar"}

    def __init__(self, api_key=None, base_url="https://api.openai.com/v1",
                 transcription_model="gpt-transcribe", speech_model="gpt-4o-mini-tts", timeout=120.0):
        self.api_key = api_key or os.getenv("AETHRION_AUDIO_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.transcription_model = transcription_model
        self.speech_model = speech_model
        self.timeout = timeout

    def _headers(self):
        if not self.api_key:
            raise AudioProviderError("Configure AETHRION_AUDIO_API_KEY or OPENAI_API_KEY.")
        return {"Authorization": f"Bearer {self.api_key}"}

    def transcribe(self, audio_path, language=None):
        path = Path(audio_path)
        if not path.is_file() or not 0 < path.stat().st_size <= self.MAX_AUDIO_BYTES:
            raise AudioProviderError("Audio file is missing, empty, or larger than 25 MiB.")
        data = {"model": self.transcription_model}
        if language:
            data["language"] = str(language)[:16]
        try:
            with path.open("rb") as stream:
                response = httpx.post(
                    self.base_url + "/audio/transcriptions", headers=self._headers(), data=data,
                    files={"file": (path.name, stream, "application/octet-stream")}, timeout=self.timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AudioProviderError(f"Transcription failed: {type(exc).__name__}") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise AudioProviderError("Speech provider returned no transcript.")
        return text.strip()

    def synthesize(self, text, output_path=None, voice="marin", instructions="Speak clearly and naturally in Turkish.", response_format="mp3"):
        if not isinstance(text, str) or not text.strip() or len(text) > self.MAX_TEXT_CHARS:
            raise AudioProviderError("Text must contain 1–4096 characters.")
        if voice not in self.VOICES:
            raise AudioProviderError("Unsupported speech voice.")
        if response_format not in {"mp3","opus","aac","flac","wav","pcm"}:
            raise AudioProviderError("Unsupported audio output format.")
        body = {"model": self.speech_model, "input": text, "voice": voice,
                "instructions": str(instructions)[:4096], "response_format": response_format}
        try:
            response = httpx.post(self.base_url + "/audio/speech", headers=self._headers(),
                                  json=body, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AudioProviderError(f"Speech synthesis failed: {type(exc).__name__}") from exc
        if not response.content or len(response.content) > 30 * 1024 * 1024:
            raise AudioProviderError("Speech provider returned an empty or oversized audio file.")
        if output_path is None:
            return response.content
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return target
