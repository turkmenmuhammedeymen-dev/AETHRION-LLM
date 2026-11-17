"""Runtime-level smoke tests for the Hugging Face Gradio Space (no provider secrets)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hf_space"))

import gradio as gr
import app as space

assert isinstance(space.demo, gr.Blocks), "Gradio Blocks did not initialize"
status = space.status()
for capability in ("chat", "code", "image", "3d", "research", "file_analysis", "audio"):
    assert capability in status, f"Missing capability status: {capability}"

assert "checkpoint/tokenizer" in space.chat("hello", [])
assert "AETHRION_CODE_API_KEY" in space.code("write a hello world function")
assert "TAVILY_API_KEY" in space.research("latest science")
assert "MESHY_API_KEY" in space.three_d_text("simple object")[0]
assert "MESHY_API_KEY" in space.three_d_image(None, "")[0]
assert "MESHY_API_KEY" in space.three_d_poll("")[0]
assert "AETHRION_CODE_API_KEY" in space.analyse(None, "summarize")
assert "AETHRION_AUDIO_API_KEY" in space.voice(None)[0]
assert "code provider" in space.edit_project(None, "review")[0].lower()

try:
    space.image("a simple landscape")
except gr.Error:
    pass
else:
    raise AssertionError("Image route should clearly report missing provider credentials")

# Exercise document create, extract and edit paths without network calls.
docx_path = space.make_doc("AETHRION Smoke Test", "first paragraph\n\nsecond paragraph", "docx")
assert Path(docx_path).is_file() and Path(docx_path).stat().st_size > 0
extracted = space.extract_doc(docx_path)
assert "first paragraph" in extracted and "second paragraph" in extracted
message, edited_path = space.edit_doc(docx_path, "first paragraph", "updated paragraph", "txt")
assert "created" in message.lower()
assert Path(edited_path).is_file()
assert "updated paragraph" in Path(edited_path).read_text(encoding="utf-8")

for fmt in ("pdf", "txt", "md", "json", "csv"):
    output = space.make_doc("Format Test", "format smoke test", fmt)
    assert Path(output).is_file() and Path(output).stat().st_size > 0, fmt

print("SPACE_IMPORT=PASS")
print("GRADIO_BLOCKS=PASS")
print("SEVEN_CAPABILITY_ROUTES=PASS")
print("NO_CREDENTIAL_FALLBACKS=PASS")
print("DOCUMENT_CREATE_EXTRACT_EDIT=PASS")
print("DOCUMENT_FORMATS=PASS")
