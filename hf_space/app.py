import os, base64, tempfile
from pathlib import Path
import gradio as gr
from aethrion.api import build_router
from aethrion.audio import OpenAISpeechProvider
from aethrion.document_tools import create_document, extract_document_text
from aethrion.file_agent import FileAnalysisAgent
from aethrion.routing import Capability, CapabilityRequest, CapabilityUnavailable
from aethrion.workspace_agent import ProjectWorkspaceAgent

router, providers = build_router()
audio = OpenAISpeechProvider() if (os.getenv("AETHRION_AUDIO_API_KEY") or os.getenv("OPENAI_API_KEY")) else None

def dispatch(capability, prompt, options=None):
    return router.dispatch(capability, CapabilityRequest(prompt, options or {})).output

def status():
    available=router.available()
    rows=[f"- **{c.value}**: {available.get(c.value, 'NOT CONFIGURED')}" for c in Capability]
    return "### Capability status\n"+"\n".join(rows)

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
        return "Configure AETHRION_CODE_API_KEY and AETHRION_CODE_MODEL."

def image(prompt):
    try:
        r=dispatch(Capability.IMAGE,prompt[:4000]); item=r.get("image",{})
        if item.get("b64_json"):
            raw=base64.b64decode(item["b64_json"],validate=True)
            if len(raw)>20*1024*1024: raise gr.Error("Image too large.")
            p=Path(tempfile.gettempdir())/("aethrion-"+os.urandom(5).hex()+".png");p.write_bytes(raw);return str(p)
        return item.get("url")
    except CapabilityUnavailable as e:
        raise gr.Error("Configure AETHRION_IMAGE_API_KEY or OPENAI_API_KEY.") from e

def three_d(prompt):
    provider=providers.get("3d")
    if not provider:return "Configure MESHY_API_KEY."
    try:
        r=provider(CapabilityRequest(prompt[:2000]))
        return f"Task {r['task_id']} created; status PENDING. It is not a completed model."
    except Exception as e:return "3D task failed: "+type(e).__name__

def research(query):
    try:return str(dispatch(Capability.RESEARCH,query[:2000],{"max_results":6}))
    except CapabilityUnavailable:return "Configure TAVILY_API_KEY."

def analyse(file_path,question):
    provider=providers.get("file_analysis")
    if not provider:return "Configure AETHRION_CODE_API_KEY and AETHRION_CODE_MODEL."
    if not file_path:return "Upload a file."
    p=Path(file_path)
    if p.stat().st_size>8*1024*1024:return "Maximum upload size: 8 MiB."
    data=base64.b64encode(p.read_bytes()).decode("ascii")
    r=FileAnalysisAgent(provider).analyze(p.name,data,question or "Summarize this file.")
    return r.get("text",str(r)) if isinstance(r,dict) else str(r)

def voice(path):
    if audio is None:return "Configure AETHRION_AUDIO_API_KEY or OPENAI_API_KEY.",None
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

def edit_project(path,task):
    provider=providers.get("file_analysis")
    if not provider:return "Configure code provider.",None
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
            p=gr.Textbox(lines=3,label="Text-to-3D prompt");o=gr.Textbox()
            gr.Button("Create 3D task").click(three_d,inputs=p,outputs=o)
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
        with gr.Tab("Audio"):
            a=gr.Audio(sources=["microphone","upload"],type="filepath")
            text=gr.Markdown();spoken=gr.Audio(type="filepath")
            gr.Button("Recognize and respond").click(voice,inputs=a,outputs=[text,spoken])
    gr.Markdown("---\nCredentials stay in Space Secrets. Uploaded project code is never executed. Training is not started from this interface.")

if __name__=="__main__":demo.queue(default_concurrency_limit=2,max_size=20).launch()
