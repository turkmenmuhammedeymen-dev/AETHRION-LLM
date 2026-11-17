"""AETHRION API with explicit local and specialist capability routing."""
from __future__ import annotations
import os
import base64
import tempfile
from pathlib import Path
from typing import Any
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter,CapabilityUnavailable
from aethrion.native_registry import NativeModelRegistry,NativeModelUnavailable,NativeModelError
from aethrion.native_runtime import NativeModelRuntime
from aethrion.workspace_agent import ProjectWorkspaceAgent, WorkspaceError
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


def _require_safety_allow(runtime: Any, text: str) -> dict[str, Any]:
    """Fail closed unless the independently released SAFE model returns ALLOW."""
    if not isinstance(text, str) or not text.strip():
        raise CapabilityUnavailable("AETHRION-SAFE received no reviewable text; request stopped.")
    try:
        if runtime.registry.status(Capability.SAFETY.value) != "ready":
            raise CapabilityUnavailable("AETHRION-SAFE is not trained and released; request stopped.")
        verdict = runtime.classify_safety(text)
    except CapabilityUnavailable:
        raise
    except Exception as exc:
        raise CapabilityUnavailable("AETHRION-SAFE inference failed; request stopped.") from exc
    decision = verdict.get("decision") if isinstance(verdict, dict) else None
    if decision != "ALLOW":
        label = decision if decision in {"REVIEW", "BLOCK"} else "UNKNOWN"
        raise CapabilityUnavailable(f"AETHRION-SAFE returned {label}; request stopped.")
    return verdict



def _require_safety_allow_http(runtime: Any, text: str) -> dict[str, Any]:
    """Translate a fail-closed SAFE verdict into a protected API response."""
    try:
        return _require_safety_allow(runtime, text)
    except CapabilityUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _text_for_safety(result: Any) -> str:
    """Extract user-visible textual output for a second safety pass."""
    if not isinstance(result, dict):
        return ""
    pieces = []
    for key in ("text", "extractive_span", "answer"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value)
    rows = result.get("results")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                for key in ("text", "passage", "snippet", "content"):
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        pieces.append(value)
    return "\n".join(pieces)


def get_native_model_statuses(manifest: str | Path | None = None) -> dict[str, str]:
    """Report verified per-model release status without loading model weights."""
    selected = Path(manifest or os.environ.get("AETHRION_MODEL_MANIFEST", "models/manifest.json"))
    registry = NativeModelRegistry(Path.cwd(), selected)
    return {name: registry.status(name) for name in sorted(registry.capabilities())}


def build_router(*, allow_remote_providers: bool = False, allow_local_model: bool = True)->tuple[CapabilityRouter,dict[str,Any]]:
    """Build a native-only router from independently trained and verified local model releases."""
    if allow_remote_providers:
        raise ValueError("AETHRION native-only policy forbids external model and search APIs.")
    router=CapabilityRouter()
    providers={}
    if not allow_local_model:
        return router,providers
    manifest=Path(os.environ.get("AETHRION_MODEL_MANIFEST","models/manifest.json"))
    try:
        runtime=NativeModelRuntime(Path.cwd(),manifest)
    except NativeModelError as exc:
        providers["runtime_error"]=type(exc).__name__
        return router,providers
    providers["runtime"]=runtime
    # The retrieval and layout/OCR adapters are not registered until their own
    # local index and document preprocessing pipelines have passed release tests.
    supported = (
        Capability.CHAT, Capability.CODE, Capability.VISION, Capability.IMAGE,
        Capability.ASR, Capability.TTS, Capability.THREE_D, Capability.RESEARCH,
        Capability.DOCUMENT, Capability.SAFETY,
    )
    safety_ready = runtime.registry.status(Capability.SAFETY.value) == "ready"
    for capability in supported:
        if runtime.registry.status(capability.value) != "ready":
            continue
        # No capability except the SAFE classifier is exposed until SAFE itself
        # has a verified local release.
        if capability != Capability.SAFETY and not safety_ready:
            continue
        def handler(request: CapabilityRequest, selected=capability.value):
            if selected != Capability.SAFETY.value:
                _require_safety_allow(runtime, request.prompt)
            result = runtime.handle(selected, request)
            if selected != Capability.SAFETY.value:
                output_text = _text_for_safety(result)
                if output_text:
                    _require_safety_allow(runtime, output_text)
            return result
        router.register(capability,runtime.registry.resolve(capability.value).architecture,handler)
    return router,providers

app=FastAPI(title="AETHRION-LLM API",version="0.4.0")
router,providers=build_router()

@app.get("/health")
def health(): return {"status":"ok","service":"AETHRION-LLM","version":"0.3.0"}

@app.get("/capabilities", dependencies=[Depends(require_api_token)])
def capabilities():
    available=router.available(); all_names=[c.value for c in Capability]
    try:
        model_status = get_native_model_statuses()
    except Exception as exc:
        model_status = {"registry": f"error:{type(exc).__name__}"}
    return {"available":available,"unavailable":[n for n in all_names if n not in available],
            "model_status":model_status,"execution_policy":"local_only","credentials_exposed":False}

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
    runtime=providers.get("runtime")
    if runtime is None or Capability.DOCUMENT.value not in router.available():
        raise HTTPException(status_code=503,detail="AETHRION-DOC is not trained and released.")
    _require_safety_allow_http(runtime, request.question)
    try:
        raw=base64.b64decode(request.content_base64,validate=True)
    except (ValueError,TypeError) as exc:
        raise HTTPException(status_code=400,detail="Invalid base64 document content.") from exc
    if not raw or len(raw)>8*1024*1024:
        raise HTTPException(status_code=413,detail="Document must be between 1 byte and 8 MiB.")
    with tempfile.NamedTemporaryFile(suffix=Path(request.filename).suffix.lower()) as handle:
        handle.write(raw); handle.flush()
        try:
            result=runtime.analyze_document(handle.name)
            _require_safety_allow_http(runtime, str(result.get("text", "")))
        except CapabilityUnavailable:
            raise HTTPException(status_code=503,detail="AETHRION-SAFE did not allow this document output.")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422,detail=f"Native document analysis failed: {type(exc).__name__}") from exc
    result["question"]=request.question
    return {"capability":"document","result":result}

@app.get("/3d/tasks/{task_id}", dependencies=[Depends(require_api_token)])
def get_3d_task(task_id:str):
    raise HTTPException(status_code=410,detail="Native 3D generation is synchronous. Use POST /3d/generate; no hosted task service is used.")

@app.post("/3d/generate", dependencies=[Depends(require_api_token)])
def generate_3d(request:dict[str,Any]):
    runtime=providers.get("runtime")
    if runtime is None or Capability.THREE_D.value not in router.available():
        raise HTTPException(status_code=503,detail="AETHRION-3D is not trained and released.")
    prompt=request.get("prompt")
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
        raise HTTPException(status_code=400,detail="prompt must contain 1–4000 characters.")
    _require_safety_allow_http(runtime, prompt)
    path=None
    try:
        path=runtime.generate_point_cloud(prompt,inference_steps=min(int(request.get("inference_steps",50)),100))
        return Response(content=path.read_bytes(),media_type="text/plain",
            headers={"Content-Disposition":'attachment; filename="aethrion-point-cloud.obj"',
                     "X-Aethrion-3D-Format":"vertex-only-obj"})
    except Exception as exc:
        raise HTTPException(status_code=422,detail=f"Native point-cloud generation failed: {type(exc).__name__}") from exc
    finally:
        if path is not None: path.unlink(missing_ok=True)

@app.post("/vision/compare", dependencies=[Depends(require_api_token)])
async def vision_compare(file:UploadFile=File(...),text:str=Form(...)):
    runtime=providers.get("runtime")
    if runtime is None or Capability.VISION.value not in router.available():
        raise HTTPException(status_code=503,detail="AETHRION-VISION is not trained and released.")
    raw=await file.read(20*1024*1024+1)
    if not raw or len(raw)>20*1024*1024:
        raise HTTPException(status_code=413,detail="Image must be between 1 byte and 20 MiB.")
    suffix=Path(file.filename or "image.png").suffix.lower()
    if suffix not in {".png",".jpg",".jpeg",".webp"}:
        raise HTTPException(status_code=400,detail="Supported image formats are PNG, JPG and WebP.")
    _require_safety_allow_http(runtime, text)
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(raw); handle.flush()
        try: score=runtime.image_text_similarity(handle.name,text)
        except Exception as exc: raise HTTPException(status_code=422,detail=f"Native image comparison failed: {type(exc).__name__}") from exc
    return {"capability":"vision","similarity":score,
            "note":"Embedding similarity only; not a natural-language image description."}

@app.post("/audio/transcriptions", dependencies=[Depends(require_api_token)])
async def audio_transcriptions(file:UploadFile=File(...)):
    runtime=providers.get("runtime")
    if runtime is None or Capability.ASR.value not in router.available():
        raise HTTPException(status_code=503,detail="AETHRION-ASR is not trained and released.")
    raw=await file.read(25*1024*1024+1)
    if not raw or len(raw)>25*1024*1024:
        raise HTTPException(status_code=413,detail="Audio must be between 1 byte and 25 MiB.")
    suffix=Path(file.filename or "audio.wav").suffix.lower()
    if suffix!=".wav":
        raise HTTPException(status_code=400,detail="Native ASR currently accepts PCM WAV only.")
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(raw); handle.flush()
        try:
            transcript=runtime.transcribe_wav(handle.name)
            _require_safety_allow_http(runtime, transcript)
        except HTTPException:
            raise
        except CapabilityUnavailable:
            raise HTTPException(status_code=503,detail="AETHRION-SAFE did not allow this transcript.")
        except Exception as exc: raise HTTPException(status_code=422,detail=f"Native ASR failed: {type(exc).__name__}") from exc
    return {"capability":"asr","text":transcript}

@app.post("/audio/speech", dependencies=[Depends(require_api_token)])
def audio_speech(request:dict[str,Any]):
    runtime=providers.get("runtime")
    if runtime is None or Capability.TTS.value not in router.available():
        raise HTTPException(status_code=503,detail="AETHRION-TTS and AETHRION-VOCODER are not trained and released.")
    text=request.get("text")
    if not isinstance(text,str) or not text.strip() or len(text)>4096:
        raise HTTPException(status_code=400,detail="text must contain 1–4096 characters.")
    _require_safety_allow(runtime, text)
    path=None
    try:
        path=runtime.synthesize_wav(text)
        return Response(content=path.read_bytes(),media_type="audio/wav",
                        headers={"Cache-Control":"no-store"})
    except Exception as exc:
        raise HTTPException(status_code=422,detail=f"Native TTS failed: {type(exc).__name__}") from exc
    finally:
        if path is not None: path.unlink(missing_ok=True)

@app.post("/documents/extract", dependencies=[Depends(require_api_token)])
async def documents_extract(file: UploadFile = File(...)):
    import tempfile
    runtime=providers.get("runtime")
    if runtime is None or runtime.registry.status(Capability.SAFETY.value) != "ready":
        raise HTTPException(status_code=503,detail="AETHRION-SAFE is not trained and released.")
    suffix=Path(file.filename or "").suffix[:12]
    raw=await file.read(8*1024*1024+1)
    if len(raw)>8*1024*1024: raise HTTPException(status_code=413,detail="Document exceeds 8 MiB.")
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(raw); handle.flush()
        try:
            content=extract_document_text(handle.name)
            _require_safety_allow_http(runtime, content)
        except CapabilityUnavailable:
            raise HTTPException(status_code=503,detail="AETHRION-SAFE did not allow this extracted document.")
        except (DocumentError,UnicodeError,ValueError) as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    return {"filename":file.filename,"text":content}

@app.post("/documents/create", dependencies=[Depends(require_api_token)])
def documents_create(request: dict[str,Any]):
    try:
        raw,name,mime=create_document(str(request.get("title","")),str(request.get("content","")),str(request.get("format","docx")))
        return Response(content=raw,media_type=mime,headers={"Content-Disposition":f'attachment; filename="{name}"',"Cache-Control":"no-store"})
    except DocumentError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.post("/workspace/propose", dependencies=[Depends(require_api_token)])
async def workspace_propose(file:UploadFile=File(...),task:str=Form(...)):
    runtime=providers.get("runtime")
    if runtime is None or Capability.CODE.value not in router.available():
        raise HTTPException(status_code=503,detail="AETHRION-CODE is not trained and released.")
    raw=await file.read(8*1024*1024+1)
    if not raw or len(raw)>8*1024*1024:
        raise HTTPException(status_code=413,detail="Project ZIP must be between 1 byte and 8 MiB.")
    _require_safety_allow_http(runtime, task)
    def local_code_provider(request):
        _require_safety_allow_http(runtime, request.prompt)
        generated = runtime.generate_text("code",request.prompt,max_new_tokens=1024,temperature=0.1)
        _require_safety_allow_http(runtime, _text_for_safety(generated))
        return generated
    try:
        output,summary=ProjectWorkspaceAgent(local_code_provider).propose(raw,task)
        return Response(content=output,media_type="application/zip",
            headers={"Content-Disposition":'attachment; filename="aethrion-review.zip"',
                     "X-Aethrion-Summary":summary[:500],"Cache-Control":"no-store"})
    except HTTPException:
        raise
    except WorkspaceError as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422,detail=f"Native project review failed: {type(exc).__name__}") from exc
