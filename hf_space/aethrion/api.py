"""AETHRION API with explicit local and specialist capability routing."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter,CapabilityUnavailable
from aethrion.native_registry import NativeModelRegistry,NativeModelUnavailable
from aethrion.document_tools import create_document,extract_document_text,DocumentError
from aethrion.security import require_api_token
from fastapi import UploadFile,File,Form,Response,Depends
try:
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,Field
except ImportError as exc: raise RuntimeError("Install API extras: pip install -e '.[api]'") from exc

class DispatchRequest(BaseModel):
    capability:str
    prompt:str=Field(min_length=1,max_length=12000)
    options:dict[str,Any]=Field(default_factory=dict)
class FileRequest(BaseModel):
    filename:str=Field(min_length=1,max_length=255)
    content_base64:str=Field(min_length=1,max_length=12_000_000)
    question:str=Field(default="Summarize and analyze this file.",max_length=2000)

def build_router(*, allow_remote_providers: bool = False, allow_local_model: bool = True)->tuple[CapabilityRouter,dict[str,Any]]:
    """Build a native-only router from hash-verified local first-party artifacts."""
    if allow_remote_providers:
        raise ValueError("AETHRION native-only policy forbids external model and search APIs.")
    router=CapabilityRouter()
    providers={}
    if allow_local_model:
        manifest=Path(os.environ.get("AETHRION_MODEL_MANIFEST","models/manifest.json"))
        registry=NativeModelRegistry(Path.cwd(),manifest)
        try:
            artifacts=registry.resolve("chat")
        except NativeModelUnavailable:
            artifacts=None
        if artifacts is not None:
            def local_chat(request:CapabilityRequest)->dict[str,str]:
                import torch
                from aethrion.chat import load_model
                from aethrion.tokenizer import ByteBPETokenizer
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model=load_model(str(artifacts.checkpoint),device)
                if artifacts.tokenizer is None:
                    raise RuntimeError("AETHRION-BASE tokenizer is not registered.")
                tokenizer=ByteBPETokenizer.load(artifacts.tokenizer)
                ids=tokenizer.encode(request.prompt)
                if not ids:
                    raise ValueError("Prompt produced no tokens.")
                context=torch.tensor([ids],dtype=torch.long,device=device)
                with torch.no_grad():
                    result=model.generate(context,min(int(request.options.get("max_new_tokens",128)),512))
                return {"text":tokenizer.decode(result[0,len(ids):].tolist()),"model":"AETHRION-BASE"}
            router.register(Capability.CHAT,"local-aethrion-base",local_chat)
    return router,providers

app=FastAPI(title="AETHRION-LLM API",version="0.4.0")
router,providers=build_router()
audio_provider=None  # Native-only policy: AETHRION-ASR/TTS must be trained locally.
code_provider=providers.get("file_analysis")

@app.get("/health")
def health(): return {"status":"ok","service":"AETHRION-LLM","version":"0.3.0"}

@app.get("/capabilities", dependencies=[Depends(require_api_token)])
def capabilities():
    available=router.available(); all_names=[c.value for c in Capability]
    return {"available":available,"unavailable":[n for n in all_names if n not in available],"credentials_exposed":False}

@app.post("/dispatch", dependencies=[Depends(require_api_token)])
def dispatch(request:DispatchRequest):
    try:
        result=router.dispatch(request.capability,CapabilityRequest(request.prompt,request.options))
        return {"capability":result.capability.value,"provider":result.provider,"result":result.output}
    except (ValueError,TypeError) as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    except CapabilityUnavailable as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=500,detail=f"Native capability operation failed: {type(exc).__name__}") from exc

@app.post("/files/analyze", dependencies=[Depends(require_api_token)])
def analyze_file(request:FileRequest):
    raise HTTPException(status_code=503,detail="AETHRION-DOC is not trained or installed. No external model fallback is permitted.")

@app.get("/3d/tasks/{task_id}", dependencies=[Depends(require_api_token)])
def get_3d_task(task_id:str):
    raise HTTPException(status_code=503,detail="AETHRION-3D is not trained or installed. No external model fallback is permitted.")

@app.post("/audio/transcriptions", dependencies=[Depends(require_api_token)])
async def audio_transcriptions(file: UploadFile = File(...), language: str | None = Form(default=None)):
    raise HTTPException(status_code=503, detail="AETHRION-ASR is not trained or installed. No external speech API is used.")

@app.post("/audio/speech", dependencies=[Depends(require_api_token)])
def audio_speech(request: dict[str,Any]):
    raise HTTPException(status_code=503, detail="AETHRION-TTS is not trained or installed. No external speech API is used.")

@app.post("/documents/extract", dependencies=[Depends(require_api_token)])
async def documents_extract(file: UploadFile = File(...)):
    import tempfile
    suffix=Path(file.filename or "").suffix[:12]
    raw=await file.read(8*1024*1024+1)
    if len(raw)>8*1024*1024: raise HTTPException(status_code=413,detail="Document exceeds 8 MiB.")
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(raw); handle.flush()
        try: content=extract_document_text(handle.name)
        except (DocumentError,UnicodeError,ValueError) as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    return {"filename":file.filename,"text":content}

@app.post("/documents/create", dependencies=[Depends(require_api_token)])
def documents_create(request: dict[str,Any]):
    try:
        raw,name,mime=create_document(str(request.get("title","")),str(request.get("content","")),str(request.get("format","docx")))
        return Response(content=raw,media_type=mime,headers={"Content-Disposition":f'attachment; filename="{name}"',"Cache-Control":"no-store"})
    except DocumentError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.post("/workspace/propose", dependencies=[Depends(require_api_token)])
async def workspace_propose(file: UploadFile = File(...), task: str = Form(...)):
    raise HTTPException(status_code=503,detail="AETHRION-CODE is not trained or installed. No external code model is used.")
