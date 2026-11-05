"""AETHRION API with explicit local and specialist capability routing."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter,CapabilityUnavailable
from aethrion.providers import MeshyTextTo3DProvider,OpenAICompatibleChatProvider,OpenAIImageProvider,TavilyResearchProvider
from aethrion.file_agent import FileAnalysisAgent,FileAnalysisError
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
    return router,providers

app=FastAPI(title="AETHRION-LLM API",version="0.3.0")
router,providers=build_router()

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
