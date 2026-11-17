import os, base64, tempfile
import httpx
from pathlib import Path
import gradio as gr
from aethrion.api import build_router
from aethrion.audio import OpenAISpeechProvider
from aethrion.document_tools import create_document, extract_document_text, replace_document_text
from aethrion.file_agent import FileAnalysisAgent
from aethrion.routing import Capability, CapabilityRequest, CapabilityUnavailable
from aethrion.workspace_agent import ProjectWorkspaceAgent

def _resolve_space_auth(environ=None):
    env = os.environ if environ is None else environ
    username = str(env.get("AETHRION_SPACE_USERNAME", "")).strip()
    password = str(env.get("AETHRION_SPACE_PASSWORD", ""))
    if not username or not password:
        return None
    return (username, password)


SPACE_AUTH = _resolve_space_auth()
REMOTE_PROVIDERS_ENABLED = SPACE_AUTH is not None
router, providers = build_router(allow_remote_providers=REMOTE_PROVIDERS_ENABLED)
audio = (
    OpenAISpeechProvider()
    if REMOTE_PROVIDERS_ENABLED and (os.getenv("AETHRION_AUDIO_API_KEY") or os.getenv("OPENAI_API_KEY"))
    else None
)


def _remote_disabled_message():
    return "Remote provider features are disabled. Configure AETHRION_SPACE_USERNAME and AETHRION_SPACE_PASSWORD in Space Secrets to enable them safely."

def dispatch(capability, prompt, options=None):
    return router.dispatch(capability, CapabilityRequest(prompt, options or {})).output

def status():
    available = router.available()
    rows = [f"- **{c.value}**: {available.get(c.value, 'NOT CONFIGURED')}" for c in Capability]
    if not REMOTE_PROVIDERS_ENABLED:
        rows.append("- **Remote providers**: DISABLED — Space authentication is not configured")
    return "### Capability status\\n" + "\\n".join(rows)

def chat(message, history):
    context=[]
    for item in (history or [])[-8:]:
        if isinstance(item,dict) and isinstance(item.get("content"),str):
            context.append(str(item.get("role","user"))+": "+item["content"][:1500])
    try:
        result=dispatch(Capability.CHAT,"\n".join(context+["user: "+message[:4000],"assistant:"]),{"max_new_tokens":256})
        return result.get("text",str(result)) if isinstance(result,dict) else str(result)
    except CapabilityUnavailable:
        return "AETHRION's own trained checkpoint/tokenizer is not loaded. No third-party model is silently substituted."

def code(prompt):
    try:
        r=dispatch(Capability.CODE,prompt[:10000],{"max_tokens":4096,"temperature":0.1})
        return r.get("text",str(r)) if isinstance(r,dict) else str(r)
    except CapabilityUnavailable:
        return _remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure AETHRION_CODE_API_KEY and AETHRION_CODE_MODEL."

def image(prompt):
    try:
        r=dispatch(Capability.IMAGE,prompt[:4000]); item=r.get("image",{})
        if item.get("b64_json"):
            raw=base64.b64decode(item["b64_json"],validate=True)
            if len(raw)>20*1024*1024: raise gr.Error("Image too large.")
            p=Path(tempfile.gettempdir())/("aethrion-"+os.urandom(5).hex()+".png");p.write_bytes(raw);return str(p)
        return item.get("url")
    except CapabilityUnavailable as e:
        message = _remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure AETHRION_IMAGE_API_KEY or OPENAI_API_KEY."
        raise gr.Error(message) from e

def three_d_text(prompt):
    provider=providers.get("3d")
    if not provider:return (_remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure MESHY_API_KEY."),""
    try:
        r=provider(CapabilityRequest(prompt[:2000]))
        task_ref="text:"+r["task_id"]
        return "Text-to-3D task created; status PENDING.",task_ref
    except Exception as e:return "3D task failed: "+type(e).__name__,""

def three_d_image(path,texture_prompt):
    provider=providers.get("3d")
    if not provider:return (_remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure MESHY_API_KEY."),""
    if not path:return "Upload a JPG, PNG or WebP reference image.",""
    try:
        r=provider.image_to_3d(path,texture_prompt or "")
        task_ref="image:"+r["task_id"]
        return "Image-to-3D task created; status PENDING.",task_ref
    except Exception as e:return "Image-to-3D failed: "+type(e).__name__,""

def three_d_poll(task_ref):
    provider=providers.get("3d")
    if not provider:return (_remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure MESHY_API_KEY."),None
    if not task_ref or ":" not in task_ref:return "Enter a task reference.",None
    kind,task_id=task_ref.split(":",1)
    try:
        result=provider.get_image_task(task_id) if kind=="image" else provider.get_task(task_id)
        state=result.get("status","UNKNOWN")
        detail="**Status:** "+str(state)
        if result.get("progress") is not None:detail+=" · **Progress:** "+str(result["progress"])+"%"
        if state=="SUCCEEDED":
            url=(result.get("model_urls") or {}).get("glb")
            if url:
                response=httpx.get(url,timeout=90,follow_redirects=True)
                response.raise_for_status()
                if len(response.content)>100*1024*1024:return detail+"\nGLB exceeds 100 MiB.",None
                path=Path(tempfile.gettempdir())/("aethrion-3d-"+os.urandom(5).hex()+".glb")
                path.write_bytes(response.content)
                return detail+"\nGLB model is ready.",str(path)
        if state in {"FAILED","CANCELED"}:detail+="\n"+str(result.get("task_error") or "Provider task failed.")
        else:detail+="\nTask is still processing; poll again later."
        return detail,None
    except Exception as e:return "Task check failed: "+type(e).__name__,None

def research(query):
    try:return str(dispatch(Capability.RESEARCH,query[:2000],{"max_results":6}))
    except CapabilityUnavailable:return _remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure TAVILY_API_KEY."

def analyse(file_path,question):
    provider=providers.get("file_analysis")
    if not provider:return _remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure AETHRION_CODE_API_KEY and AETHRION_CODE_MODEL."
    if not file_path:return "Upload a file."
    p=Path(file_path)
    if p.stat().st_size>8*1024*1024:return "Maximum upload size: 8 MiB."
    data=base64.b64encode(p.read_bytes()).decode("ascii")
    r=FileAnalysisAgent(provider).analyze(p.name,data,question or "Summarize this file.")
    return r.get("text",str(r)) if isinstance(r,dict) else str(r)

def voice(path):
    if audio is None:return (_remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure AETHRION_AUDIO_API_KEY or OPENAI_API_KEY."),None
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
    provider=providers.get("file_analysis")
    if not provider:return _remote_disabled_message() if not REMOTE_PROVIDERS_ENABLED else "Configure code provider.",None
    if not path:return "Upload a project ZIP.",None
    raw,summary=ProjectWorkspaceAgent(provider).propose(Path(path).read_bytes(),task)
    p=Path(tempfile.gettempdir())/("aethrion-review-"+os.urandom(5).hex()+".zip");p.write_bytes(raw)
    return summary+"\n\nReview the ZIP; uploaded code was not executed.",str(p)

with gr.Blocks(title="AETHRION-LLM · Intelligence Without Limits",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AETHRION-LLM\n### Intelligence Without Limits")
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
