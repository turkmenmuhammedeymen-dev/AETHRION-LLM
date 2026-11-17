"""AETHRION API with explicit local and specialist capability routing."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter,CapabilityUnavailable
from aethrion.providers import MeshyTextTo3DProvider,OpenAICompatibleChatProvider,OpenAIImageProvider,TavilyResearchProvider
from aethrion.file_agent import FileAnalysisAgent,FileAnalysisError
from aethrion.audio import OpenAISpeechProvider,AudioProviderError
from aethrion.document_tools import create_document,extract_document_text,DocumentError
from aethrion.workspace_agent import ProjectWorkspaceAgent,WorkspaceError
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
    """Build a native-only router. Remote inference providers are forbidden."""
    if allow_remote_providers:
        raise ValueError("AETHRION native-only policy forbids external model and search APIs.")
    router=CapabilityRouter()
    providers={}
    checkpoint=Path(os.environ.get("AETHRION_CHECKPOINT","artifacts/checkpoint_last.pt"))
    tokenizer_path=Path(os.environ.get("AETHRION_TOKENIZER","artifacts/tokenizer.json"))
    if allow_local_model and checkpoint.is_file() and tokenizer_path.is_file():
        def local_chat(request:CapabilityRequest)->dict[str,str]:
            import torch
            from aethrion.chat import load_model
            from aethrion.tokenizer import ByteBPETokenizer
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model=load_model(str(checkpoint),device)
            tokenizer=ByteBPETokenizer.load(tokenizer_path)
            ids=tokenizer.encode(request.prompt)
            if not ids:
                raise ValueError("Prompt produced no tokens.")
            context=torch.tensor([ids],dtype=torch.long,device=device)
            with torch.no_grad():
                result=model.generate(context,min(int(request.options.get("max_new_tokens",128)),512))
            return {"text":tokenizer.decode(result[0,len(ids):].tolist()),"model":"AETHRIONLM"}
        router.register(Capability.CHAT,"local-aethrionlm",local_chat)
    return router,providers

app=FastAPI(title="AETHRION-LLM API",version="0.4.0")
router,providers=build_router()
audio_key=os.environ.get("AETHRION_AUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY")
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
    except Exception as exc: raise HTTPException(status_code=502,detail=f"Provider operation failed: {type(exc).__name__}") from exc

@app.post("/files/analyze", dependencies=[Depends(require_api_token)])
def analyze_file(request:FileRequest):
    provider=providers.get("file_analysis")
    if provider is None: raise HTTPException(status_code=503,detail="AETHRION-DOC is not trained or installed; no external model fallback is allowed.")
    try:
        result=FileAnalysisAgent(provider).analyze(request.filename,request.content_base64,request.question)
        return {"capability":"file_analysis","provider":"openai-compatible-file-agent","result":result}
    except FileAnalysisError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=502,detail=f"File analysis failed: {type(exc).__name__}") from exc

@app.get("/3d/tasks/{task_id}", dependencies=[Depends(require_api_token)])
def get_3d_task(task_id:str):
    provider=providers.get("3d")
    if provider is None: raise HTTPException(status_code=503,detail="AETHRION-3D is not trained or installed; no external model fallback is allowed.")
    try: return provider.get_task(task_id)
    except Exception as exc: raise HTTPException(status_code=502,detail=f"3D task lookup failed: {type(exc).__name__}") from exc


@app.post("/audio/transcriptions", dependencies=[Depends(require_api_token)])
async def audio_transcriptions(file: UploadFile = File(...), language: str | None = Form(default=None)):
    if audio_provider is None:
        raise HTTPException(status_code=503, detail="AETHRION-ASR is not trained or installed; no external speech API is used.")
    import tempfile
    suffix=Path(file.filename or "audio.bin").suffix[:12] or ".bin"
    raw=await file.read(OpenAISpeechProvider.MAX_AUDIO_BYTES+1)
    if not raw or len(raw)>OpenAISpeechProvider.MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413,detail="Audio exceeds 25 MiB.")
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(raw); handle.flush()
        try: text=audio_provider.transcribe(handle.name,language)
        except AudioProviderError as exc: raise HTTPException(status_code=502,detail=str(exc)) from exc
    return {"text":text}

@app.post("/audio/speech", dependencies=[Depends(require_api_token)])
def audio_speech(request: dict[str,Any]):
    if audio_provider is None:
        raise HTTPException(status_code=503,detail="AETHRION-TTS is not trained or installed; no external speech API is used.")
    try:
        raw=audio_provider.synthesize(str(request.get("text","")),voice=str(request.get("voice","marin")))
        return Response(content=raw,media_type="audio/mpeg",headers={"Cache-Control":"no-store"})
    except AudioProviderError as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc

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
    if code_provider is None:
        raise HTTPException(status_code=503,detail="AETHRION-CODE is not trained or installed; no external code model is used.")
    raw=await file.read(8*1024*1024+1)
    if len(raw)>8*1024*1024: raise HTTPException(status_code=413,detail="Project ZIP exceeds 8 MiB.")
    try:
        output,summary=ProjectWorkspaceAgent(code_provider).propose(raw,task)
        return Response(content=output,media_type="application/zip",headers={"Content-Disposition":'attachment; filename="aethrion-project-proposal.zip"',"X-Aethrion-Summary":summary[:500],"Cache-Control":"no-store"})
    except WorkspaceError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
