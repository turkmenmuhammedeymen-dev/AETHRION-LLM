import base64,unittest
from unittest.mock import patch
from aethrion.file_agent import FileAnalysisError,extract_text
from aethrion.providers import MeshyTextTo3DProvider,OpenAICompatibleChatProvider,OpenAIImageProvider,ProviderError,TavilyResearchProvider
from aethrion.routing import CapabilityRequest

class ProviderAdapterTests(unittest.TestCase):
    def test_chat_completion(self):
        provider=OpenAICompatibleChatProvider("key","model","https://example.test/v1")
        with patch("aethrion.providers._request_json",return_value={"choices":[{"message":{"content":"Merhaba"}}]}):
            self.assertEqual(provider(CapabilityRequest("Selam"))["text"],"Merhaba")
    def test_chat_rejects_missing_content(self):
        provider=OpenAICompatibleChatProvider("key","model","https://example.test/v1")
        with patch("aethrion.providers._request_json",return_value={"choices":[]}):
            with self.assertRaises(ProviderError): provider(CapabilityRequest("x"))
    def test_image_requires_payload(self):
        provider=OpenAIImageProvider("key")
        with patch("aethrion.providers._request_json",return_value={"data":[{"url":"https://example.test/a.png"}]}):
            self.assertIn("image",provider(CapabilityRequest("landscape")))
        with patch("aethrion.providers._request_json",return_value={"data":[]}):
            with self.assertRaises(ProviderError): provider(CapabilityRequest("landscape"))
    def test_meshy_returns_async_task(self):
        provider=MeshyTextTo3DProvider("key")
        with patch("aethrion.providers._request_json",return_value={"result":"task-1"}):
            result=provider(CapabilityRequest("chair"))
        self.assertEqual(result,{"provider":"meshy","task_id":"task-1","status":"PENDING"})
    def test_tavily_results(self):
        provider=TavilyResearchProvider("key")
        with patch("aethrion.providers._request_json",return_value={"results":[{"url":"https://example.test"}]}):
            self.assertEqual(len(provider(CapabilityRequest("query"))["results"]),1)
        with patch("aethrion.providers._request_json",return_value={}):
            with self.assertRaises(ProviderError): provider(CapabilityRequest("query"))
    def test_text_and_json_extraction(self):
        txt=base64.b64encode("Türkçe".encode()).decode()
        self.assertEqual(extract_text("note.txt",txt),"Türkçe")
        js=base64.b64encode(b'{"ok":true}').decode()
        self.assertIn('"ok": true',extract_text("data.json",js))
    def test_file_limits_and_extensions(self):
        data=base64.b64encode(b"data").decode()
        with self.assertRaises(FileAnalysisError): extract_text("../secret.exe",data)
        huge=base64.b64encode(b"x"*(8*1024*1024+1)).decode()
        with self.assertRaises(FileAnalysisError): extract_text("large.txt",huge)

if __name__=="__main__": unittest.main()
