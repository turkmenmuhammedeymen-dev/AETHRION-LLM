import os, base64, tempfile
import spaces
from pathlib import Path
import gradio as gr
from fastapi import HTTPException
from aethrion.api import build_router, get_native_model_statuses, _require_safety_allow_http, _require_safety_allow_chunks_http
from aethrion.document_tools import create_document, extract_document_text, replace_document_text
from aethrion.routing import Capability, CapabilityRequest, CapabilityUnavailable
from aethrion.workspace_agent import MAX_ARCHIVE_BYTES, ProjectWorkspaceAgent, WorkspaceError, read_project_zip
from aethrion.zerogpu_preflight import run_zero_gpu_preflight
from aethrion.space_auth import space_auth_required
from aethrion.space_training import render_training_readiness

def _resolve_space_auth(environ=None):
    env = os.environ if environ is None else environ
    username = str(env.get("AETHRION_SPACE_USERNAME", "")).strip()
    password = str(env.get("AETHRION_SPACE_PASSWORD", ""))
    if not username or not password:
        return None
    return (username, password)


SPACE_AUTH = _resolve_space_auth()
SPACE_AUTH_ENABLED = SPACE_AUTH is not None
# Production Space is native-only: no hosted inference/search/model APIs are enabled.
REMOTE_PROVIDERS_ENABLED = False
router, providers = build_router(allow_remote_providers=False, allow_local_model=SPACE_AUTH_ENABLED)
audio = None


def _remote_disabled_message():
    return "This AETHRION-native specialist model is not trained or installed yet. No external model or API fallback is used."

def _auth_required_message():
    return "Local AETHRION model access is disabled. Configure Space authentication before enabling private local inference."

def dispatch(capability, prompt, options=None):
    return router.dispatch(capability, CapabilityRequest(prompt, options or {})).output


@spaces.GPU(duration=1)
def _zerogpu_registration_noop():
    # Registered as a hidden, non-API event so ZeroGPU recognizes this Space.
    # It is never invoked by normal UI usage and consumes no GPU quota.
    return None


@spaces.GPU(duration=5)
def gpu_diagnostics():
    """Report the GPU visible inside an actual ZeroGPU request, without loading models."""
    try:
        import torch
    except Exception as exc:
        return "### ZeroGPU diagnostics\n- PyTorch import failed: " + type(exc).__name__
    if not torch.cuda.is_available():
        return (
            "### ZeroGPU diagnostics\n"
            "- GPU request completed, but CUDA is not visible in this execution.\n"
            "- No model was loaded and no training was started."
        )
    try:
        device_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        total_gib = properties.total_memory / (1024 ** 3)
        free_gib, _ = torch.cuda.mem_get_info(0)
        free_gib /= 1024 ** 3
        return (
            "### ZeroGPU diagnostics\n"
            f"- CUDA: **available**\n"
            f"- Device: **{device_name}**\n"
            f"- Device memory: **{total_gib:.2f} GiB total**\n"
            f"- Currently free: **{free_gib:.2f} GiB**\n"
            "- This was a short diagnostic request only; no training was started."
        )
    except Exception as exc:
        return (
            "### ZeroGPU diagnostics\n"
            "- CUDA is available, but device details could not be read: "
            + type(exc).__name__
        )


@spaces.GPU(duration=60)
def _gpu_local_chat(prompt):
    result = dispatch(Capability.CHAT, prompt, {"max_new_tokens": 256})
    return result.get("text", str(result)) if isinstance(result, dict) else str(result)


@spaces.GPU(duration=60)
def gpu_model_preflight():
    """Exercise every native architecture with disposable synthetic tensors only."""
    try:
        report = run_zero_gpu_preflight()
    except Exception as exc:
        return "### Native model GPU preflight failed\n" + type(exc).__name__ + ": " + str(exc)[:300]
    lines = [
        "### AETHRION native-model ZeroGPU preflight",
        f"- **GPU:** {report['device']}",
        f"- **VRAM:** {report['total_memory_gib']} GiB total",
        f"- **Peak allocated:** {report['peak_allocated_gib']} GiB",
        f"- **Architectures:** {report['models_passed']}/{report['models_tested']} passed",
        "",
        "| Model | Result | Parameters | Synthetic loss |",
        "|---|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['model']} | {row['status']} | {row.get('parameters', 0):,} | {row.get('synthetic_loss', '—')} |"
        )
    for row in report["results"]:
        if row.get("error"):
            lines.append(f"\n**{row['model']} error:** `{row['error']}`")
    lines.extend([
        "",
        "**Safety:** synthetic tensors only; no dataset read, no optimizer step, no checkpoint written, no production training started.",
        f"**Overall:** {report['status']} — all models remain NOT_RELEASED.",
    ])
    return "\n".join(lines)


def training_readiness():
    """Expose non-sensitive, read-only data-gate status without model access."""
    try:
        return render_training_readiness(Path(__file__).resolve())
    except Exception as exc:
        return "### Training readiness unavailable\n- " + type(exc).__name__ + ": " + str(exc)[:240]

TRAIN_SESSION_MODELS = ["chat", "code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document"]


def training_setup_status():
    """Booleans only: never reveal secret values."""
    token = bool(os.environ.get("HF_TOKEN"))
    repo = bool(os.environ.get("AETHRION_ARTIFACT_REPO", "").strip())
    lines = ["### Training session setup",
             f"- Space authentication: {'OK' if SPACE_AUTH_ENABLED else 'MISSING'}",
             f"- HF_TOKEN secret: {'OK' if token else 'MISSING'}",
             f"- AETHRION_ARTIFACT_REPO variable: {'OK' if repo else 'MISSING'}"]
    if not (SPACE_AUTH_ENABLED and token and repo):
        lines.append("Sessions stay disabled until all three are configured (see docs/SETUP_PRIVATE_ARTIFACT_REPO.md).")
    return "\n".join(lines)


@spaces.GPU(duration=110)
def _gpu_training_session(model, steps):
    from aethrion.space_session import run_session, graceful_runner
    return run_session(model, int(steps), root=Path(__file__).resolve().parent, timeout=85,
                       runner=graceful_runner, prepare_chat=True)


def training_session(model, steps):
    """One bounded, resumable, gate-checked session. Never releases a model."""
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    if model not in TRAIN_SESSION_MODELS:
        return "Only models with a validated data gate can run: " + ", ".join(TRAIN_SESSION_MODELS)
    if not os.environ.get("HF_TOKEN") or not os.environ.get("AETHRION_ARTIFACT_REPO", "").strip():
        return training_setup_status()
    try:
        result = _gpu_training_session(model, steps)
    except Exception as exc:
        return "### Training session failed\n- " + type(exc).__name__ + ": " + str(exc)[:300]
    progress = result.get("progress", {})
    trainer_ok = bool(result.get("trainer_ok"))
    complete = bool(progress.get("complete"))
    if not trainer_ok:
        heading = "### Training session failed"
    elif complete:
        heading = "### Training session completed (not released)"
    else:
        heading = "### Training session paused / incomplete (not released)"
    return (
        heading + "\n"
        + f"- Model: {result.get('model', model)} | steps requested: {result.get('steps', steps)}\n"
        + f"- Trainer ok: {trainer_ok} | exit code: {result.get('trainer_exit_code', 'unknown')} | pulled: {len(result.get('pulled', []))} files | pushed: {len(result.get('pushed', []))} files\n"
        + f"- Progress: {progress.get('status', 'unknown')} step {progress.get('step')} | complete: {complete}\n"
        + "- Released: NO — evaluation and human approval are still required.\n"
        + f"```\n{result.get('trainer_tail', result.get('note', 'No diagnostic available.'))}\n```"
    )


def status():
    try:
        model_status = get_native_model_statuses()
    except Exception as exc:
        model_status = {"registry": "ERROR — " + type(exc).__name__}
    rows = [f"- **{name}**: {state.upper()}" for name, state in model_status.items()]
    if not SPACE_AUTH_ENABLED:
        rows.append("- **Local model access**: DISABLED — Space authentication is not configured")
    rows.append("- **External model/API providers**: DISABLED — native-only policy")
    return "### Native model release status\n" + "\n".join(rows)

def _chat_core(message, history):
    context = []
    for item in (history or [])[-8:]:
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            context.append(str(item.get("role", "user")) + ": " + item["content"][:1500])
    prompt = "\n".join(context + ["user: " + message[:4000], "assistant:"])
    try:
        return _gpu_local_chat(prompt)
    except CapabilityUnavailable:
        return "AETHRION's own trained checkpoint/tokenizer is not available. No third-party model is silently substituted."
# Private preview is separate from the production router and release registry.
_EXPERIMENTAL_CHAT_MODEL = None
_EXPERIMENTAL_CHAT_TOKENIZER = None
_EXPERIMENTAL_CHAT_INFO = None


def chat(message, history):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    from aethrion.chat_orchestrator import orchestrate
    think = False
    for prefix in ("/düşün ", "/think "):
        if message.startswith(prefix):
            think, message = True, message[len(prefix):]
            break
    trace: list[str] = []
    answer = orchestrate(
        message, router.available(),
        lambda capability, prompt: dispatch(capability, prompt),
        lambda prompt: _chat_core(prompt, history),
        trace=trace if think else None,
    )
    if think and trace:  # thinking mode: show the real steps taken, not invented reasoning
        answer = "Düşünme adımları:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(trace, 1)) + "\n\n" + answer
    return answer


@spaces.GPU(duration=60)
def _gpu_experimental_chat(prompt):
    global _EXPERIMENTAL_CHAT_MODEL, _EXPERIMENTAL_CHAT_TOKENIZER, _EXPERIMENTAL_CHAT_INFO
    from aethrion.experimental_chat import load_experimental_chat, generate_experimental_chat
    if _EXPERIMENTAL_CHAT_MODEL is None:
        root = Path(__file__).resolve().parent
        _EXPERIMENTAL_CHAT_MODEL, _EXPERIMENTAL_CHAT_TOKENIZER, _EXPERIMENTAL_CHAT_INFO = load_experimental_chat(root)
    answer = generate_experimental_chat(
        _EXPERIMENTAL_CHAT_MODEL, _EXPERIMENTAL_CHAT_TOKENIZER, prompt,
        max_new_tokens=64,
    )
    return answer, _EXPERIMENTAL_CHAT_INFO


def experimental_chat(message, history):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    if not isinstance(message, str) or not message.strip():
        return "Enter a prompt for the experimental Chat candidate."
    if len(message) > 2000:
        return "Experimental preview prompts are limited to 2,000 characters."
    context = []
    for item in (history or [])[-4:]:
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            context.append(str(item.get("role", "user")) + ": " + item["content"][:500])
    prompt = "\\n".join(context + ["user: " + message, "assistant:"])
    try:
        answer, info = _gpu_experimental_chat(prompt)
        return (
            "**EXPERIMENTAL CANDIDATE — NOT RELEASED**\\n\\n"
            + answer
            + "\\n\\n---\\n"
            + f"Checkpoint step: {info.get('step')} · Device: {info.get('device')}\\n"
            + "This private preview bypasses release registration only; it does not change "
              "models/manifest.json, approval flags, or production routing."
        )
    except Exception as exc:
        return "Experimental Chat load/inference failed: " + type(exc).__name__ + ": " + str(exc)[:240]


def code(prompt):
    if space_auth_required(SPACE_AUTH_ENABLED):
        return _auth_required_message()
    try:
        r=dispatch(Capability.CODE,prompt[:10000],{"max_tokens":4096,"temperature":0.1})
        return r.get("text",str(r)) if isinstance(r,dict) else str(r)
    except CapabilityUnavailable:
        return _auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-CODE is not trained or installed. No external code model is used."

@spaces.GPU(duration=60)
def vision_compare(image_path, text):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    if not image_path:
        return "Upload an image to compare."
    if not isinstance(text, str) or not text.strip():
        return "Enter text to compare with the image."
    try:
        result = dispatch(
            Capability.VISION, text[:4000], {"image_path": str(image_path)}
        )
        score = float(result.get("similarity"))
        return (
            f"**Image/text embedding similarity:** {score:.4f}\n\n"
            "This score compares representations; it is not a natural-language "
            "image description or visual question-answering result."
        )
    except CapabilityUnavailable:
        return "AETHRION-VISION is not trained and released yet. No external vision model is used."
    except Exception as exc:
        return "Native vision comparison failed: " + type(exc).__name__


@spaces.GPU(duration=60)
def image(prompt):
    if not SPACE_AUTH_ENABLED:
        raise gr.Error(_auth_required_message())
    try:
        result = dispatch(Capability.IMAGE, prompt[:4000], {"inference_steps": 50})
        path = result.get("image_path") if isinstance(result, dict) else None
        if not path or not Path(path).is_file():
            raise gr.Error("AETHRION-IMAGE returned no local PNG file.")
        return path
    except CapabilityUnavailable as exc:
        raise gr.Error("AETHRION-IMAGE is unavailable until local visual-output safety screening is implemented and released.") from exc
    except Exception as exc:
        raise gr.Error("Native image generation failed: " + type(exc).__name__) from exc

@spaces.GPU(duration=60)
def three_d_text(prompt):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message(), ""
    try:
        result = dispatch(Capability.THREE_D, prompt[:2000], {"inference_steps": 50})
        path = result.get("asset_path") if isinstance(result, dict) else None
        if not path or not Path(path).is_file():
            return "AETHRION-3D returned no local OBJ file.", ""
        return "Native AETHRION point cloud generated. This is a vertex-only OBJ, not a watertight mesh.", str(path)
    except CapabilityUnavailable:
        return "AETHRION-3D is unavailable until local 3D-output safety screening is implemented and released. No external 3D service is used.", ""
    except Exception as exc:
        return "Native 3D generation failed: " + type(exc).__name__, ""

def three_d_image(path,texture_prompt):
    return "Current AETHRION-3D release architecture supports text-conditioned point clouds only; image-conditioned 3D is not implemented.", ""

@spaces.GPU(duration=60)
def research(query):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    try:
        result = dispatch(Capability.RESEARCH, query[:4000], {"top_k": 5})
        rows = result.get("results", []) if isinstance(result, dict) else []
        if not rows:
            return "No matching passages were found in the locally approved AETHRION research index."
        lines = ["## Local AETHRION research results"]
        for row in rows:
            lines.append(
                f"### {row.get('title', 'Untitled')}\n"
                f"{row.get('excerpt', '')}\n"
                f"Source: {row.get('source_uri', 'unknown')}\n"
                f"License: {row.get('license', 'unknown')}\n"
                f"Relevance: {row.get('score', 0):.4f}"
            )
        return "\n\n".join(lines)
    except CapabilityUnavailable:
        return "AETHRION-RESEARCH is not trained and released, or its local approved index is not installed. No hosted search API is used."
    except Exception as exc:
        return "Local research failed: " + type(exc).__name__

@spaces.GPU(duration=60)
def analyse(file_path,question):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    if not file_path:
        return "Upload a PDF, DOCX or supported text file."
    source=Path(file_path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size>8*1024*1024:
        return "Document upload must be a regular file no larger than 8 MiB."
    try:
        result = dispatch(
            Capability.DOCUMENT,
            str(question or "Analyze the document and identify key information."),
            {"document_path": str(file_path)}
        )
        return (
            f"**AETHRION-DOC result**\n\n{result.get('extractive_span', '')}"
            f"\n\n**Limitations:** {result.get('limitations', '')}"
            f"\n\n**Extracted text:**\n{result.get('text', '')[:12000]}"
        )
    except CapabilityUnavailable:
        return "AETHRION-DOC is not trained and released yet."
    except Exception as exc:
        return "Native document analysis failed: " + type(exc).__name__

@spaces.GPU(duration=60)
def voice(path):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message(), None
    if not path:
        return "Record or upload PCM WAV audio.", None
    try:
        recognized = dispatch(Capability.ASR, "Transcribe the provided WAV audio.", {"audio_path": str(path)})
        transcript = recognized.get("text", "")
        if not transcript:
            return "AETHRION-ASR returned an empty transcript.", None
        answer = dispatch(Capability.CHAT, transcript, {"max_new_tokens": 192}).get("text", "")
        spoken = dispatch(Capability.TTS, answer, {})
        audio_path = spoken.get("audio_path")
        return (
            f"**Transcript:** {transcript}\n\n**AETHRION-BASE:** {answer}",
            audio_path
        )
    except CapabilityUnavailable:
        return "AETHRION-ASR, AETHRION-BASE, AETHRION-TTS or AETHRION-VOCODER is not trained and released yet.", None
    except Exception as exc:
        return "Native speech conversation failed: " + type(exc).__name__, None

def make_doc(title,content,fmt):
    if not SPACE_AUTH_ENABLED:
        raise gr.Error(_auth_required_message())
    runtime=providers.get("runtime")
    if runtime is None:
        raise gr.Error("Document creation is unavailable until the local SAFE model is released.")
    if not isinstance(title,str) or not isinstance(content,str) or not isinstance(fmt,str):
        raise gr.Error("Title, content and format must be text.")
    if len(title)+len(content)>20000:
        raise gr.Error("Document text exceeds the 20,000-character review limit.")
    try:
        review_text=title+"\n"+content
        _require_safety_allow_http(runtime,review_text)
        _require_safety_allow_chunks_http(runtime,review_text)
        raw,name,mime=create_document(title,content,fmt)
        _require_safety_allow_chunks_http(runtime,content)
        output=Path(tempfile.gettempdir())/("aethrion-"+os.urandom(5).hex()+"-"+name)
        output.write_bytes(raw)
        return str(output)
    except HTTPException as exc:
        raise gr.Error("Document creation blocked by AETHRION-SAFE: "+str(exc.detail)) from exc
    except Exception as exc:
        raise gr.Error("Document creation failed: "+type(exc).__name__) from exc

def extract_doc(path):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    if not path:
        return "Upload a document."
    runtime=providers.get("runtime")
    if runtime is None:
        return "Document extraction is unavailable until the local SAFE model is released."
    source=Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size>8*1024*1024:
        return "Document upload must be a regular file no larger than 8 MiB."
    try:
        text=extract_document_text(source)
        _require_safety_allow_chunks_http(runtime,text)
        return text
    except HTTPException as exc:
        return "Document extraction blocked by AETHRION-SAFE: "+str(exc.detail)
    except Exception as exc:
        return "Document extraction failed: "+type(exc).__name__

def edit_doc(path,find_text,replacement,fmt):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message(),None
    if not path:
        return "Upload a document.",None
    runtime=providers.get("runtime")
    if runtime is None:
        return "Document editing is unavailable until the local SAFE model is released.",None
    if not isinstance(find_text,str) or not isinstance(replacement,str) or not isinstance(fmt,str):
        return "Find text, replacement and format must be text.",None
    if len(find_text)+len(replacement)>20000:
        return "Document edit exceeds the 20,000-character review limit.",None
    output=None
    try:
        source_text=extract_document_text(path)
        _require_safety_allow_chunks_http(runtime,source_text)
        _require_safety_allow_http(runtime,replacement)
        raw,name,mime=replace_document_text(path,find_text,replacement,fmt)
        output=Path(tempfile.gettempdir())/("aethrion-edited-"+os.urandom(5).hex()+"-"+name)
        output.write_bytes(raw)
        edited_text=extract_document_text(output)
        _require_safety_allow_chunks_http(runtime,edited_text)
        return "Edited copy created. Original file was not modified.",str(output)
    except HTTPException as exc:
        if output is not None:
            output.unlink(missing_ok=True)
        return "Document editing blocked by AETHRION-SAFE: "+str(exc.detail),None
    except Exception as exc:
        if output is not None:
            output.unlink(missing_ok=True)
        return "Document editing failed: "+type(exc).__name__,None

@spaces.GPU(duration=60)
def edit_project(path,task):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message(), None
    if not path:
        return "Upload a project ZIP.", None
    if not task or not task.strip():
        return "Enter a project change request.", None
    runtime = providers.get("runtime")
    try:
        _require_safety_allow_http(runtime, task)
        source_file=Path(path)
        if not source_file.is_file() or source_file.is_symlink():
            raise WorkspaceError("Project upload must be a regular file.")
        if source_file.stat().st_size>MAX_ARCHIVE_BYTES:
            raise WorkspaceError("Project ZIP exceeds the 8 MiB upload limit.")
        project_bytes = source_file.read_bytes()
        for filename, source_text in read_project_zip(project_bytes).items():
            _require_safety_allow_chunks_http(runtime, source_text)
        def local_code_provider(request):
            return dispatch(Capability.CODE, request.prompt, {"max_new_tokens": 4096, "temperature": 0.1})
        raw, summary = ProjectWorkspaceAgent(local_code_provider).propose(
            project_bytes, task
        )
        # Review every generated file, not only the agent's natural-language summary.
        for filename, output_text in read_project_zip(raw).items():
            _require_safety_allow_chunks_http(runtime, output_text)
        _require_safety_allow_chunks_http(runtime, summary)
        output = Path(tempfile.gettempdir()) / (
            "aethrion-review-" + os.urandom(5).hex() + ".zip"
        )
        output.write_bytes(raw)
        return summary + "\n\nUploaded code was never executed; review the ZIP before applying it.", str(output)
    except HTTPException as exc:
        return "Project rejected by AETHRION-SAFE: " + str(exc.detail), None
    except CapabilityUnavailable:
        return "AETHRION-CODE is not trained and released yet.", None
    except WorkspaceError as exc:
        return "Project proposal rejected: " + str(exc), None
    except Exception as exc:
        return "Project editing failed: " + type(exc).__name__, None



with gr.Blocks(title="AETHRION-LLM · Intelligence Without Limits",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AETHRION-LLM\n### Intelligence Without Limits")
    _zero_gpu_registration = gr.Button(visible=False, interactive=False)
    _zero_gpu_registration.click(
        fn=_zerogpu_registration_noop,
        inputs=[],
        outputs=[],
        api_name=False,
    )
    s=gr.Markdown(status());gr.Button("Refresh status").click(status,outputs=s)
    gpu_report=gr.Markdown("ZeroGPU status has not been checked in this session.")
    gr.Button("Check ZeroGPU hardware").click(gpu_diagnostics,outputs=gpu_report)
    preflight_report=gr.Markdown("Native architecture preflight has not been run in this session.")
    gr.Button("Run 11-model ZeroGPU preflight").click(gpu_model_preflight,outputs=preflight_report)
    with gr.Accordion("11-model training readiness", open=False):
        gr.Markdown(
            "This check validates the real local corpus and specialist manifests. "
            "It is read-only: it does not start an optimizer, create checkpoints, "
            "or change any training approval."
        )
        readiness_report = gr.Markdown("Training data gates have not been checked in this session.")
        gr.Button("Check all 11 model data gates (read-only)").click(
            training_readiness, inputs=[], outputs=readiness_report
        )
    with gr.Accordion("Training session (owner only)", open=False):
        gr.Markdown(
            "Runs one short, resumable, data-gate-checked session for a single model and mirrors "
            "its checkpoint to your private artifact repo. It cannot release a model."
        )
        setup_report = gr.Markdown("Setup has not been checked in this session.")
        gr.Button("Check training setup (booleans only)").click(training_setup_status, inputs=[], outputs=setup_report)
        train_model = gr.Dropdown(TRAIN_SESSION_MODELS, value="code", label="Model")
        train_steps = gr.Slider(1, 500, value=200, step=1, label="Optimizer steps this session")
        train_report = gr.Markdown()
        gr.Button("Run one training session").click(training_session, inputs=[train_model, train_steps], outputs=train_report)
    with gr.Tabs():
        with gr.Tab("AI Chat"):
            gr.ChatInterface(chat,type="messages",chatbot=gr.Chatbot(type="messages",height=420))
        if SPACE_AUTH_ENABLED:
            with gr.Tab("Experimental Chat · owner only"):
                gr.Markdown(
                    "**Private experimental candidate. Not released, not safety-reviewed, and not the production model route.** "
                    "The checkpoint and tokenizer hashes are checked before loading. Output is limited to 64 new tokens."
                )
                gr.ChatInterface(
                    experimental_chat, type="messages",
                    chatbot=gr.Chatbot(type="messages", height=420),
                )
        with gr.Tab("Code & Project"):
            p=gr.Textbox(lines=4,label="Code task");o=gr.Code(label="Generated code")
            gr.Button("Generate").click(code,inputs=p,outputs=o)
            z=gr.File(label="Project ZIP",file_types=[".zip"],type="filepath")
            t=gr.Textbox(lines=3,label="Requested project changes")
            sm=gr.Markdown();d=gr.File(label="Review ZIP")
            gr.Button("Propose project changes").click(edit_project,inputs=[z,t],outputs=[sm,d])
        with gr.Tab("Image"):
            gr.Markdown(
                "Image generation is disabled until a local visual-output safety verifier "
                "is implemented, tested and released."
            )
            gr.Button("Generate image — unavailable", interactive=False)
        with gr.Tab("Vision"):
            vision_image=gr.Image(type="filepath",sources=["upload"],label="Upload image")
            vision_text=gr.Textbox(lines=2,label="Text to compare with the image")
            vision_result=gr.Markdown()
            gr.Button("Compare image and text").click(
                vision_compare,inputs=[vision_image,vision_text],outputs=vision_result
            )
        with gr.Tab("3D Studio"):
            gr.Markdown(
                "3D generation, image-to-3D conversion and asset downloads are disabled. "
                "They will remain unavailable until local geometry-output safety screening "
                "and a server-owned, access-controlled artifact store are implemented, "
                "tested and released."
            )
            gr.Button("Create from text — unavailable", interactive=False)
            gr.Button("Image-to-3D — unavailable", interactive=False)
        with gr.Tab("Research"):
            q=gr.Textbox(lines=3,label="Research question");o=gr.Markdown()
            gr.Button("Search").click(research,inputs=q,outputs=o)
        with gr.Tab("Files & Documents"):
            f=gr.File(type="filepath",label="TXT / MD / JSON / CSV / LOG / PDF / DOCX")
            q=gr.Textbox(label="Analysis request",value="Summarize and list key points.")
            o=gr.Markdown();gr.Button("Analyse").click(analyse,inputs=[f,q],outputs=o)
            gr.Markdown("#### Create document")
            title=gr.Textbox(label="Title");content=gr.Textbox(lines=8,label="Content")
            fmt=gr.Dropdown(["docx","pdf","txt","md","json","csv"],value="docx")
            out=gr.File();gr.Button("Create").click(make_doc,inputs=[title,content,fmt],outputs=out)
            src=gr.File(type="filepath",label="Extract document")
            extracted=gr.Textbox(lines=8);gr.Button("Extract").click(extract_doc,inputs=src,outputs=extracted)
            gr.Markdown("#### Edit text in an existing document")
            source_edit=gr.File(type="filepath",label="Source document")
            find_text=gr.Textbox(label="Find exact text")
            replacement=gr.Textbox(label="Replace with")
            edit_fmt=gr.Dropdown(["docx","pdf","txt","md","json","csv"],value="docx",label="Edited copy format")
            edit_status=gr.Markdown();edit_download=gr.File(label="Edited copy")
            gr.Button("Create edited copy").click(edit_doc,inputs=[source_edit,find_text,replacement,edit_fmt],outputs=[edit_status,edit_download])
        with gr.Tab("Audio"):
            a=gr.Audio(sources=["microphone","upload"],type="filepath")
            text=gr.Markdown();spoken=gr.Audio(type="filepath")
            gr.Button("Recognize and respond").click(voice,inputs=a,outputs=[text,spoken])
    gr.Markdown("---\nCredentials stay in Space Secrets. Uploaded project code is never executed. Training starts only from the owner-only session panel, after every data gate passes.")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        prevent_thread_lock=False,
        ssr_mode=False,
        auth=SPACE_AUTH,
        show_error=True,
    )
