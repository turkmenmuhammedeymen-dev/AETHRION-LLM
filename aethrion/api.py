"""AETHRION API with explicit local and specialist capability routing."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter,CapabilityUnavailable
from aethrion.providers import MeshyTextTo3DProvider,OpenAICompatibleChatProvider,OpenAIImageProvider,TavilyResearchProvider
from aethrion.file_agent import FileAnalysisAgent,FileAnalysisError
from aethrion.audio import OpenAISpeechProvider,AudioProviderError\nfrom aethrion.document_tools import create_document,extract_document_text,DocumentError\nfrom aethrion.workspace_agent import ProjectWorkspaceAgent,WorkspaceError\nfrom fastapi import UploadFile,File,Form,Response
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

def build_router()->tuple[CapabilityRouter,dict[str,Any]]:
    router=CapabilityRouter(); providers={}
    code_key=os.environ.get("AETHRION_CODE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    code_model=os.environ.get("AETHRION_CODE_MODEL")
    code_base=os.environ.get("AETHRION_CODE_BASE_URL","https://api.openai.com/v1")
    if code_key and code_model:
        code=OpenAICompatibleChatProvider(code_key,code_model,code_base)
        router.register(Capability.CODE,"openai-compatible-code",code)
        router.register(Capability.FILE_ANALYSIS,"openai-compatible-file-agent",code)
        providers["file_analysis"]=code
    image_key=os.environ.get("AETHRION_IMAGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if image_key:
        router.register(Capability.IMAGE,"openai-images",OpenAIImageProvider(
            image_key,os.environ.get("AETHRION_IMAGE_MODEL","gpt-image-2"),
            os.environ.get("AETHRION_IMAGE_BASE_URL","https://api.openai.com/v1")))
    meshy_key=os.environ.get("MESHY_API_KEY")
    if meshy_key:
        meshy=MeshyTextTo3DProvider(meshy_key,os.environ.get("AETHRION_MESHY_BASE_URL","https://api.meshy.ai/openapi/v2"))
        router.register(Capability.THREE_D,"meshy-text-to-3d",meshy); providers["3d"]=meshy
    tavily_key=os.environ.get("TAVILY_API_KEY")
    if tavily_key: router.register(Capability.RESEARCH,"tavily-search",TavilyResearchProvider(tavily_key))
    checkpoint=Path(os.environ.get("AETHRION_CHECKPOINT","artifacts/checkpoint_last.pt"))
    tokenizer_path=Path(os.environ.get("AETHRION_TOKENIZER","artifacts/tokenizer.json"))
    if checkpoint.is_file() and tokenizer_path.is_file():
        def local_chat(request:CapabilityRequest)->dict[str,str]:
            import torch
            from aethrion.chat import load_model
            from aethrion.tokenizer import ByteBPETokenizer
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model=load_model(str(checkpoint),device); tokenizer=ByteBPETokenizer.load(tokenizer_path)
            ids=tokenizer.encode(request.prompt)
            if not ids: raise ValueError("Prompt produced no tokens.")
            context=torch.tensor([ids],dtype=torch.long,device=device)
            with torch.no_grad(): result=model.generate(context,min(int(request.options.get("max_new_tokens",128)),512))
            return {"text":tokenizer.decode(result[0,len(ids):].tolist()),"model":"AETHRIONLM"}
        router.register(Capability.CHAT,"local-aethrionlm",local_chat)
    audio_key=os.environ.get("AETHRION_AUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if audio_key:
        from aethrion.audio import OpenAISpeechProvider
        import base64
        speech=OpenAISpeechProvider(api_key=audio_key)
        def audio_handler(request:CapabilityRequest)->dict[str,Any]:
            raw=speech.synthesize(request.prompt,voice=str(request.options.get("voice","marin")))
            return {"format":"mp3","audio_base64":base64.b64encode(raw).decode("ascii")}
        router.register(Capability.AUDIO,"openai-turn-based-speech",audio_handler)
    return router,providers

app=FastAPI(title="AETHRION-LLM API",version="0.4.0")
router,providers=build_router()
audio_key=os.environ.get("AETHRION_AUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY")
audio_provider=OpenAISpeechProvider(api_key=audio_key) if audio_key else None
code_provider=providers.get("file_analysis")

@app.get("/health")
def health(): return {"status":"ok","service":"AETHRION-LLM","version":"0.3.0"}

@app.get("/capabilities")
def capabilities():
    available=router.available(); all_names=[c.value for c in Capability]
    return {"available":available,"unavailable":[n for n in all_names if n not in available],"credentials_exposed":False}

@app.post("/dispatch")
def dispatch(request:DispatchRequest):
    try:
        result=router.dispatch(request.capability,CapabilityRequest(request.prompt,request.options))
        return {"capability":result.capability.value,"provider":result.provider,"result":result.output}
    except (ValueError,TypeError) as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    except CapabilityUnavailable as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=502,detail=f"Provider operation failed: {type(exc).__name__}") from exc

@app.post("/files/analyze")
def analyze_file(request:FileRequest):
    provider=providers.get("file_analysis")
    if provider is None: raise HTTPException(status_code=503,detail="Configure AETHRION_CODE_API_KEY and AETHRION_CODE_MODEL.")
    try:
        result=FileAnalysisAgent(provider).analyze(request.filename,request.content_base64,request.question)
        return {"capability":"file_analysis","provider":"openai-compatible-file-agent","result":result}
    except FileAnalysisError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=502,detail=f"File analysis failed: {type(exc).__name__}") from exc

@app.get("/3d/tasks/{task_id}")
def get_3d_task(task_id:str):
    provider=providers.get("3d")
    if provider is None: raise HTTPException(status_code=503,detail="MESHY_API_KEY is not configured.")
    try: return provider.get_task(task_id)
    except Exception as exc: raise HTTPException(status_code=502,detail=f"3D task lookup failed: {type(exc).__name__}") from exc


@app.post("/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...), language: str | None = Form(default=None)):
    if audio_provider is None:
        raise HTTPException(status_code=503, detail="Configure AETHRION_AUDIO_API_KEY or OPENAI_API_KEY.")
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

@app.post("/audio/speech")
def audio_speech(request: dict[str,Any]):
    if audio_provider is None:
        raise HTTPException(status_code=503,detail="Configure AETHRION_AUDIO_API_KEY or OPENAI_API_KEY.")
    try:
        raw=audio_provider.synthesize(str(request.get("text","")),voice=str(request.get("voice","marin")))
        return Response(content=raw,media_type="audio/mpeg",headers={"Cache-Control":"no-store"})
    except AudioProviderError as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.post("/documents/extract")
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

@app.post("/documents/create")
def documents_create(request: dict[str,Any]):
    try:
        raw,name,mime=create_document(str(request.get("title","")),str(request.get("content","")),str(request.get("format","docx")))
        return Response(content=raw,media_type=mime,headers={"Content-Disposition":f'attachment; filename="{name}"',"Cache-Control":"no-store"})
    except DocumentError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.post("/workspace/propose")
async def workspace_propose(file: UploadFile = File(...), task: str = Form(...)):
    if code_provider is None:
        raise HTTPException(status_code=503,detail="Configure AETHRION_CODE_API_KEY and AETHRION_CODE_MODEL.")
    raw=await file.read(8*1024*1024+1)
    if len(raw)>8*1024*1024: raise HTTPException(status_code=413,detail="Project ZIP exceeds 8 MiB.")
    try:
        output,summary=ProjectWorkspaceAgent(code_provider).propose(raw,task)
        return Response(content=output,media_type="application/zip",headers={"Content-Disposition":'attachment; filename="aethrion-project-proposal.zip"',"X-Aethrion-Summary":summary[:500],"Cache-Control":"no-store"})
    except WorkspaceError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
