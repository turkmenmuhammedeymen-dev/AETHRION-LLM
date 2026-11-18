"""Fail-closed structural and semantic gate for generated visual artifacts.

Structural validation is not semantic safety. A separately trained, released,
local visual verifier must implement the runtime protocol before any artifact
can be returned to a caller.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class VisualSafetyError(RuntimeError):
    """Visual artifact is malformed, unsupported, or not approved."""


def inspect_visual_artifact(path: str | Path, modality: str) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file() or artifact.is_symlink():
        raise VisualSafetyError("Visual artifact must be a regular, non-symlink file.")
    size = artifact.stat().st_size
    if size < 1 or size > 20 * 1024 * 1024:
        raise VisualSafetyError("Visual artifact must be between 1 byte and 20 MiB.")

    if modality == "image":
        try:
            from PIL import Image
            with Image.open(artifact) as image:
                if image.format != "PNG":
                    raise VisualSafetyError("Generated image must be PNG.")
                width, height = image.size
                if not (1 <= width <= 4096 and 1 <= height <= 4096 and width * height <= 16_000_000):
                    raise VisualSafetyError("Generated image dimensions exceed safety bounds.")
                image.verify()
            return {"modality": "image", "format": "PNG", "width": width, "height": height, "bytes": size}
        except VisualSafetyError:
            raise
        except Exception as exc:
            raise VisualSafetyError("Generated image failed decoding or integrity validation.") from exc

    if modality == "3d":
        vertices = 0
        try:
            with artifact.open("r", encoding="utf-8", errors="strict") as stream:
                for line in stream:
                    record = line.strip()
                    if not record or record.startswith("#"):
                        continue
                    fields = record.split()
                    if fields[0] != "v":
                        raise VisualSafetyError("Only vertex records and comments are supported in point-cloud OBJ files.")
                    if len(fields) != 4:
                        raise VisualSafetyError("3D vertex must contain exactly three coordinates.")
                    coords = [float(value) for value in fields[1:]]
                    if not all(math.isfinite(value) and abs(value) <= 1_000_000 for value in coords):
                        raise VisualSafetyError("3D vertex contains non-finite or out-of-range coordinates.")
                    vertices += 1
                    if vertices > 1_000_000:
                        raise VisualSafetyError("3D artifact exceeds the vertex limit.")
        except VisualSafetyError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise VisualSafetyError("3D artifact is not a valid bounded UTF-8 point cloud.") from exc
        if vertices < 1:
            raise VisualSafetyError("3D artifact contains no vertices.")
        return {"modality": "3d", "format": "vertex-only-obj", "vertices": vertices, "bytes": size}

    raise VisualSafetyError("Unsupported visual modality.")


def require_visual_output_allow(runtime: Any, path: str | Path, modality: str) -> dict[str, Any]:
    """Require both structural checks and a separately released local semantic verifier."""
    inspection = inspect_visual_artifact(path, modality)
    status = getattr(runtime, "visual_safety_status", None)
    review = getattr(runtime, "review_visual_artifact", None)
    if not callable(status) or not callable(review) or status() != "ready":
        raise VisualSafetyError("AETHRION visual safety verifier is not trained and released.")
    try:
        verdict = review(str(path), modality, inspection)
    except Exception as exc:
        raise VisualSafetyError("Local visual safety inference failed; artifact blocked.") from exc
    if not isinstance(verdict, dict):
        raise VisualSafetyError("Visual safety verifier returned an invalid result.")
    confidence = verdict.get("confidence")
    if (
        verdict.get("decision") != "ALLOW"
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.90 <= float(confidence) <= 1.0
    ):
        raise VisualSafetyError("Visual safety verifier did not return a high-confidence ALLOW.")
    return {"inspection": inspection, "verdict": verdict}
