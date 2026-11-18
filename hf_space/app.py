import os, base64, tempfile
import spaces
from pathlib import Path
import gradio as gr
from fastapi import HTTPException
from aethrion.api import build_router, get_native_model_statuses, _require_safety_allow_http, _require_safety_allow_chunks_http
from aethrion.document_tools import create_document, extract_document_text, replace_document_text
from aethrion.routing import Capability, CapabilityRequest, CapabilityUnavailable
from aethrion.workspace_agent import ProjectWorkspaceAgent, WorkspaceError, read_project_zip
from aethrion.zerogpu_preflight import run_zero_gpu_preflight

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
    lines.extend([
        "",
        "**Safety:** synthetic tensors only; no dataset read, no optimizer step, no checkpoint written, no production training started.",
        f"**Overall:** {report['status']} — all models remain NOT_RELEASED.",
    ])
    return "\n".join(lines)


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

def chat(message, history):
    if not SPACE_AUTH_ENABLED:
        return _auth_required_message()
    if Capability.CHAT.value not in router.available():
        return "AETHRION's own trained checkpoint/tokenizer is not loaded. No third-party model is silently substituted."
    context = []
    for item in (history or [])[-8:]:
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            context.append(str(item.get("role", "user")) + ": " + item["content"][:1500])
    prompt = "\n".join(context + ["user: " + message[:4000], "assistant:"])
    try:
        return _gpu_local_chat(prompt)
    except CapabilityUnavailable:
        return "AETHRION's own trained checkpoint/tokenizer is not available. No third-party model is silently substituted."
@spaces.GPU(duration=60)
def code(prompt):
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

def three_d_poll(asset_path):
    if asset_path and Path(asset_path).is_file():
        return "Local OBJ point cloud is ready. It contains vertices only; no faces or textures.", asset_path
    return "No local OBJ asset found. Generate a text-conditioned point cloud first.", None

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
    raw,name,mime=create_document(title,content,fmt)
    p=Path(tempfile.gettempdir())/("aethrion-"+os.urandom(5).hex()+"-"+name);p.write_bytes(raw);return str(p)

def extract_doc(path):
    return extract_document_text(path) if path else "Upload a document."

def edit_doc(path,find_text,replacement,fmt):
    if not path:return "Upload a document.",None
    raw,name,mime=replace_document_text(path,find_text,replacement,fmt)
    p=Path(tempfile.gettempdir())/("aethrion-edited-"+os.urandom(5).hex()+"-"+name)
    p.write_bytes(raw)
    return "Edited copy created. Original file was not modified.",str(p)

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
        project_bytes = Path(path).read_bytes()
        for filename, source_text in read_project_zip(project_bytes).items():
            _require_safety_allow_chunks_http(runtime, source_text)
        def local_code_provider(request):
            return dispatch(Capability.CODE, request.prompt, {"max_new_tokens": 4096, "temperature": 0.1})
        raw, summary = ProjectWorkspaceAgent(local_code_provider).propose(
            project_bytes, task
        )
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
    with gr.Tabs():
        with gr.Tab("AI Chat"):
            gr.ChatInterface(chat,type="messages",chatbot=gr.Chatbot(type="messages",height=420))
        with gr.Tab("Code & Project"):
            p=gr.Textbox(lines=4,label="Code task");o=gr.Code(label="Generated code")
            gr.Button("Generate").click(code,inputs=p,outputs=o)
            z=gr.File(label="Project ZIP",file_types=[".zip"],type="filepath")
            t=gr.Textbox(lines=3,label="Requested project changes")
            sm=gr.Markdown();d=gr.File(label="Review ZIP")
            gr.Button("Propose project changes").click(edit_project,inputs=[z,t],outputs=[sm,d])
        with gr.Tab("Image"):
            p=gr.Textbox(lines=3,label="Image prompt");o=gr.Image(type="filepath")
            gr.Button("Generate image").click(image,inputs=p,outputs=o)
        with gr.Tab("Vision"):
            vision_image=gr.Image(type="filepath",sources=["upload"],label="Upload image")
            vision_text=gr.Textbox(lines=2,label="Text to compare with the image")
            vision_result=gr.Markdown()
            gr.Button("Compare image and text").click(
                vision_compare,inputs=[vision_image,vision_text],outputs=vision_result
            )
        with gr.Tab("3D Studio"):
            p=gr.Textbox(lines=3,label="Text-to-3D prompt");task_msg=gr.Markdown()
            task_ref=gr.Textbox(label="Local OBJ file path")
            gr.Button("Create from text").click(three_d_text,inputs=p,outputs=[task_msg,task_ref])
            image_ref=gr.Image(type="filepath",sources=["upload"],label="Image-to-3D reference")
            texture=gr.Textbox(label="Optional texture prompt")
            gr.Button("Image-to-3D unavailable — not implemented", interactive=False)
            task_status=gr.Markdown();model_file=gr.File(label="Generated OBJ point cloud (vertices only)")
            gr.Button("Check / download OBJ point cloud").click(three_d_poll,inputs=task_ref,outputs=[task_status,model_file])
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
    gr.Markdown("---\nCredentials stay in Space Secrets. Uploaded project code is never executed. Training is not started from this interface.")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        prevent_thread_lock=False,
        ssr_mode=False,
        auth=SPACE_AUTH,
        show_error=True,
    )
