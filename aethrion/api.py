"""Optional local-only FastAPI inference endpoint."""
from __future__ import annotations

import os

import torch

from aethrion.chat import load_model
from aethrion.tokenizer import ByteBPETokenizer

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError("Install API extras: pip install -e '.[api]'") from exc

CHECKPOINT = os.environ.get("AETHRION_CHECKPOINT", "artifacts/checkpoint_last.pt")
TOKENIZER_PATH = os.environ.get("AETHRION_TOKENIZER", "artifacts/tokenizer.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(CHECKPOINT, DEVICE)
tokenizer = ByteBPETokenizer.load(TOKENIZER_PATH)
app = FastAPI(title="AETHRION-0.1", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)


@app.get("/health")
def health():
    return {"status": "ok", "model": "AETHRION-0.1", "device": str(DEVICE)}


@app.post("/generate")
def generate(request: GenerateRequest):
    ids = tokenizer.encode(request.prompt)
    if not ids:
        raise HTTPException(status_code=400, detail="Prompt tokenization returned no tokens")
    context = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    try:
        with torch.no_grad():
            output = model.generate(context, request.max_new_tokens,
                                    temperature=request.temperature)
        return {"text": tokenizer.decode(output[0, len(ids):].tolist())}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Inference failed; check available memory.") from exc
