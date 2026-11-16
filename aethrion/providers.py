"""Opt-in remote provider adapters. Credentials are read only from environment variables."""
from __future__ import annotations
import json, os, urllib.error, urllib.parse, urllib.request
from typing import Any
from aethrion.routing import CapabilityRequest

class ProviderError(RuntimeError):
    pass

def _request_json(url: str, *, method: str="POST", payload: dict[str, Any] | None=None,
                  api_key: str, timeout: int=90) -> dict[str, Any]:
    if not api_key:
        raise ProviderError("Provider API key is not configured.")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers={"Authorization":f"Bearer {api_key}","Accept":"application/json"}
    if body is not None: headers["Content-Type"]="application/json"
    req=urllib.request.Request(url,data=body,headers=headers,method=method)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            raw=response.read(12*1024*1024+1)
        if len(raw)>12*1024*1024: raise ProviderError("Provider response exceeded 12 MiB.")
        data=json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail=exc.read(2048).decode("utf-8",errors="replace")
        raise ProviderError(f"Provider HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as exc:
        raise ProviderError(f"Provider request failed: {type(exc).__name__}") from exc
    if not isinstance(data,dict): raise ProviderError("Provider returned non-object JSON.")
    return data

class OpenAICompatibleChatProvider:
    def __init__(self,api_key:str,model:str,base_url:str,timeout:int=90):
        self.api_key,self.model,self.base_url,self.timeout=api_key,model,base_url.rstrip("/"),timeout
        if not all((api_key,model,self.base_url)): raise ValueError("Chat provider needs key, model and base URL.")
    def __call__(self,request:CapabilityRequest)->dict[str,Any]:
        data=_request_json(self.base_url+"/chat/completions",api_key=self.api_key,timeout=self.timeout,
            payload={"model":self.model,"messages":[{"role":"user","content":request.prompt}],
                     "temperature":float(request.options.get("temperature",0.2)),
                     "max_tokens":min(int(request.options.get("max_tokens",2048)),4096)})
        try: text=data["choices"][0]["message"]["content"]
        except (KeyError,IndexError,TypeError) as exc: raise ProviderError("Missing chat completion content.") from exc
        if not isinstance(text,str) or not text.strip(): raise ProviderError("Chat provider returned empty text.")
        return {"text":text,"model":data.get("model",self.model)}

class OpenAIImageProvider:
    def __init__(self,api_key:str,model:str="gpt-image-2",base_url:str="https://api.openai.com/v1"):
        self.api_key,self.model,self.base_url=api_key,model,base_url.rstrip("/")
    def __call__(self,request:CapabilityRequest)->dict[str,Any]:
        data=_request_json(self.base_url+"/images/generations",api_key=self.api_key,timeout=180,
            payload={"model":self.model,"prompt":request.prompt,"n":1,"size":"1024x1024"})
        items=data.get("data")
        if not isinstance(items,list) or not items or not isinstance(items[0],dict):
            raise ProviderError("Image provider returned no image data.")
        item=items[0]
        if not item.get("url") and not item.get("b64_json"): raise ProviderError("Image response has no URL or base64 payload.")
        return {"model":self.model,"image":item}

class MeshyTextTo3DProvider:
    def __init__(self,api_key:str,base_url:str="https://api.meshy.ai/openapi/v2"):
        self.api_key,self.base_url=api_key,base_url.rstrip("/")
    def __call__(self,request:CapabilityRequest)->dict[str,Any]:
        data=_request_json(self.base_url+"/text-to-3d",api_key=self.api_key,timeout=90,
            payload={"mode":"preview","prompt":request.prompt,
                     "ai_model":os.environ.get("AETHRION_MESHY_MODEL","meshy-7.1"),
                     "target_formats":["glb"]})
        task_id=data.get("result") or data.get("id")
        if not isinstance(task_id,str) or not task_id: raise ProviderError("Meshy returned no task identifier.")
        return {"provider":"meshy","task_id":task_id,"status":"PENDING"}
    def get_task(self,task_id:str)->dict[str,Any]:
        if not task_id or "/" in task_id or ".." in task_id: raise ValueError("Invalid task identifier.")
        return _request_json(self.base_url+"/text-to-3d/"+urllib.parse.quote(task_id,safe=""),
                             method="GET",api_key=self.api_key,timeout=45)

class TavilyResearchProvider:
    def __init__(self,api_key:str,base_url:str="https://api.tavily.com"):
        self.api_key,self.base_url=api_key,base_url.rstrip("/")
    def __call__(self,request:CapabilityRequest)->dict[str,Any]:
        data=_request_json(self.base_url+"/search",api_key=self.api_key,timeout=45,
            payload={"api_key":self.api_key,"query":request.prompt,"search_depth":"basic",
                     "max_results":min(int(request.options.get("max_results",5)),10),"include_answer":False})
        if not isinstance(data.get("results"),list): raise ProviderError("Research provider returned no results list.")
        return {"provider":"tavily","results":data["results"]}
