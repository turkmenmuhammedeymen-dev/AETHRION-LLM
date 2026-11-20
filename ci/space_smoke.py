"""Runtime-level smoke tests for the Hugging Face Gradio Space (no provider secrets)."""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
import sys

# Mirror the deployment job: the root package is the source of truth, while
# hf_space/aethrion may be a stale checked-in snapshot. Stage the exact package
# that deployment uploads before importing the Space entrypoint.
REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_ROOT = REPO_ROOT / "hf_space"
SPACE_PACKAGE = SPACE_ROOT / "aethrion"
if SPACE_PACKAGE.exists():
    shutil.rmtree(SPACE_PACKAGE)
shutil.copytree(REPO_ROOT / "aethrion", SPACE_PACKAGE)
sys.path.insert(0, str(SPACE_ROOT))

import gradio as gr
import app as space


assert isinstance(space.demo, gr.Blocks), "Gradio Blocks did not initialize"
status = space.status()
assert "Native model release status" in status
assert "External model/API providers" in status and "DISABLED" in status
assert "native-only" in status.lower()

old_auth, old_providers = space.SPACE_AUTH_ENABLED, space.providers


class _SmokeSafetyRegistry:
    def status(self, model_id):
        return "ready" if model_id == "safety" else "not_trained"


class _SmokeSafetyRuntime:
    registry = _SmokeSafetyRegistry()

    def __init__(self):
        self.reviewed = []

    def classify_safety(self, text):
        self.reviewed.append(text)
        return {
            "decision": "BLOCK" if "BLOCK_ME" in text else "ALLOW",
            "confidence": 0.99,
        }


try:
    # Verify that the unauthenticated Space fails closed and never asks for
    # third-party provider credentials.
    space.SPACE_AUTH_ENABLED = False
    disabled = space._auth_required_message()
    public_readiness = space.training_readiness()
    assert "11-model training readiness" in public_readiness
    assert "Models checked:** 11" in public_readiness
    assert "Training started:** NO" in public_readiness
    assert "Checkpoints written:** NO" in public_readiness
    assert public_readiness.count(" | BLOCKED | ") + public_readiness.count(" | READY — NOT STARTED | ") == 11

    assert "authentication" in disabled.lower()
    assert disabled == space.chat("hello", [])
    assert disabled == space.code("write a hello world function")
    assert disabled == space.research("local science")
    assert disabled == space.analyse(None, "summarize")
    assert disabled == space.voice(None)[0]
    assert disabled == space.three_d_text("simple object")[0]
    assert disabled == space.edit_project(None, "review")[0]

    try:
        space.image("a simple landscape")
    except gr.Error as exc:
        assert "authentication" in str(exc).lower()
    else:
        raise AssertionError("Image route must require Space authentication")

    image_3d_message, _ = space.three_d_image(None, "")
    assert "not implemented" in image_3d_message.lower()

    # Exercise the SAFE gate and document routes with an isolated in-memory
    # test runtime. This does not enable a production model or external API.
    space.SPACE_AUTH_ENABLED = True
    readiness = space.training_readiness()
    assert "Models checked:** 11" in readiness
    assert "Training started:** NO" in readiness
    assert "Checkpoints written:** NO" in readiness
    assert readiness.count(" | BLOCKED | ") + readiness.count(" | READY — NOT STARTED | ") == 11
    smoke_runtime = _SmokeSafetyRuntime()
    space.providers = {"runtime": smoke_runtime}

    with tempfile.TemporaryDirectory() as folder:
        archive_path = Path(folder) / "project.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("main.py", "# BLOCK_ME\\nprint('hello')\\n")
        rejected, output = space.edit_project(str(archive_path), "review this project")
        assert "rejected by AETHRION-SAFE" in rejected
        assert output is None
        assert any("BLOCK_ME" in segment for segment in smoke_runtime.reviewed)

        docx_path = space.make_doc(
            "AETHRION Smoke Test",
            "first paragraph\\n\\nsecond paragraph",
            "docx",
        )
        assert Path(docx_path).is_file() and Path(docx_path).stat().st_size > 0
        extracted = space.extract_doc(docx_path)
        assert "first paragraph" in extracted and "second paragraph" in extracted

        message, edited_path = space.edit_doc(
            docx_path, "first paragraph", "updated paragraph", "txt"
        )
        assert "created" in message.lower()
        assert Path(edited_path).is_file()
        assert "updated paragraph" in Path(edited_path).read_text(encoding="utf-8")

        for fmt in ("pdf", "txt", "md", "json", "csv"):
            output_path = space.make_doc("Format Test", "format smoke test", fmt)
            assert Path(output_path).is_file() and Path(output_path).stat().st_size > 0, fmt
finally:
    space.SPACE_AUTH_ENABLED, space.providers = old_auth, old_providers

print("SPACE_IMPORT=PASS")
print("GRADIO_BLOCKS=PASS")
print("NATIVE_ONLY_STATUS=PASS")
print("UNAUTHENTICATED_FAIL_CLOSED=PASS")
print("NO_THIRD_PARTY_CREDENTIAL_FALLBACKS=PASS")
print("GRADIO_PROJECT_SAFE_GATE=PASS")
print("DOCUMENT_CREATE_EXTRACT_EDIT=PASS")
print("DOCUMENT_FORMATS=PASS")
