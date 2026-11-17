import os, base64, tempfile
import spaces
from pathlib import Path
import gradio as gr
from aethrion.api import build_router, get_native_model_statuses
from aethrion.document_tools import create_document, extract_document_text, replace_document_text
from aethrion.routing import Capability, CapabilityRequest, CapabilityUnavailable

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


@spaces.GPU(duration=60)
def _gpu_local_chat(prompt):
    result = dispatch(Capability.CHAT, prompt, {"max_new_tokens": 256})
    return result.get("text", str(result)) if isinstance(result, dict) else str(result)


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
def code(prompt):
    try:
        r=dispatch(Capability.CODE,prompt[:10000],{"max_tokens":4096,"temperature":0.1})
        return r.get("text",str(r)) if isinstance(r,dict) else str(r)
    except CapabilityUnavailable:
        return _auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-CODE is not trained or installed. No external code model is used."

def image(prompt):
    try:
        r=dispatch(Capability.IMAGE,prompt[:4000]); item=r.get("image",{})
        if item.get("b64_json"):
            raw=base64.b64decode(item["b64_json"],validate=True)
            if len(raw)>20*1024*1024: raise gr.Error("Image too large.")
            p=Path(tempfile.gettempdir())/("aethrion-"+os.urandom(5).hex()+".png");p.write_bytes(raw);return str(p)
        return item.get("url")
    except CapabilityUnavailable as e:
        message = _auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-IMAGE is not trained or installed. No external image model is used."
        raise gr.Error(message) from e

def three_d_text(prompt):
    provider=providers.get("3d")
    if not provider:return (_auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-3D is not trained or installed. No external 3D model is used."),""
    try:
        r=provider(CapabilityRequest(prompt[:2000]))
        task_ref="text:"+r["task_id"]
        return "Text-to-3D task created; status PENDING.",task_ref
    except Exception as e:return "3D task failed: "+type(e).__name__,""

def three_d_image(path,texture_prompt):
    provider=providers.get("3d")
    if not provider:return (_auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-3D is not trained or installed. No external 3D model is used."),""
    if not path:return "Upload a JPG, PNG or WebP reference image.",""
    try:
        r=provider.image_to_3d(path,texture_prompt or "")
        task_ref="image:"+r["task_id"]
        return "Image-to-3D task created; status PENDING.",task_ref
    except Exception as e:return "Image-to-3D failed: "+type(e).__name__,""

def three_d_poll(task_ref):
    return "AETHRION-3D is not trained or installed. No external 3D model or API is used.", None

def research(query):
    try:return str(dispatch(Capability.RESEARCH,query[:2000],{"max_results":6}))
    except CapabilityUnavailable:return _auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-RESEARCH is not trained or its local index is not built. No external search API is used."

def analyse(file_path,question):
    return "AETHRION-DOC is not trained or installed. Local deterministic text extraction remains available."

def voice(path):
    if audio is None:return (_auth_required_message() if not SPACE_AUTH_ENABLED else "AETHRION-ASR and AETHRION-TTS are not trained or installed. No external speech API is used."),None
    if not path:return "Record or upload audio.",None
    try:
        transcript=audio.transcribe(path,"tr")
        try:
            r=dispatch(Capability.CHAT,transcript,{"max_new_tokens":192})
            answer=r.get("text",str(r)) if isinstance(r,dict) else str(r)
        except CapabilityUnavailable:answer="Speech recognized; AETHRION's own chat model is not loaded."
        raw=audio.synthesize(answer)
        p=Path(tempfile.gettempdir())/("aethrion-voice-"+os.urandom(5).hex()+".mp3");p.write_bytes(raw)
        return "**Transcript:** "+transcript+"\n\n**Answer:** "+answer,str(p)
    except Exception as e:return "Audio failed: "+type(e).__name__,None

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

def edit_project(path,task):
    return "AETHRION-CODE is not trained or installed. No external code model is used.", None


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
        with gr.Tab("3D Studio"):
            p=gr.Textbox(lines=3,label="Text-to-3D prompt");task_msg=gr.Markdown()
            task_ref=gr.Textbox(label="3D task reference")
            gr.Button("Create from text").click(three_d_text,inputs=p,outputs=[task_msg,task_ref])
            image_ref=gr.Image(type="filepath",sources=["upload"],label="Image-to-3D reference")
            texture=gr.Textbox(label="Optional texture prompt")
            gr.Button("Create from image").click(three_d_image,inputs=[image_ref,texture],outputs=[task_msg,task_ref])
            task_status=gr.Markdown();model_file=gr.File(label="Generated GLB")
            gr.Button("Check task / download GLB").click(three_d_poll,inputs=task_ref,outputs=[task_status,model_file])
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
