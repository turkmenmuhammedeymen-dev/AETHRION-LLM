import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from aethrion import api
from aethrion.api import build_router
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter

class ApiRegistrationTests(unittest.TestCase):
    def test_empty_environment_registers_no_remote_provider(self):
        env={"AETHRION_CODE_API_KEY":"","OPENAI_API_KEY":"","AETHRION_CODE_MODEL":"",
             "AETHRION_IMAGE_API_KEY":"","MESHY_API_KEY":"","TAVILY_API_KEY":"",
             "AETHRION_CHECKPOINT":"/missing/checkpoint.pt","AETHRION_TOKENIZER":"/missing/tokenizer.json"}
        with patch.dict("os.environ",env,clear=False):
            router,providers=build_router()
        self.assertEqual(router.available(),{}); self.assertEqual(providers,{})
    def test_code_and_research_register_from_env(self):
        env={"AETHRION_CODE_API_KEY":"dummy","AETHRION_CODE_MODEL":"dummy-model",
             "AETHRION_CODE_BASE_URL":"https://example.test/v1","AETHRION_IMAGE_API_KEY":"",
             "OPENAI_API_KEY":"","MESHY_API_KEY":"","TAVILY_API_KEY":"dummy",
             "AETHRION_CHECKPOINT":"/missing/checkpoint.pt","AETHRION_TOKENIZER":"/missing/tokenizer.json"}
        with patch.dict("os.environ",env,clear=False):
            router,providers=build_router()
        self.assertIn(Capability.CODE.value,router.available())
        self.assertIn(Capability.RESEARCH.value,router.available())
        self.assertIn(Capability.FILE_ANALYSIS.value,router.available())
        self.assertIn("file_analysis",providers)

class ApiHttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.router=CapabilityRouter()
        self.router.register(Capability.CODE,"test-code",lambda req:{"text":"generated code"})
        self.client=TestClient(api.app)
        self.patch=patch.object(api,"router",self.router)
        self.patch.start()
        self.addCleanup(self.patch.stop)
    def test_capabilities_endpoint_reports_registered_handlers(self):
        response=self.client.get("/capabilities")
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["available"],{"code":"test-code"})
        self.assertIn("image",response.json()["unavailable"])
    def test_dispatch_endpoint_routes_selected_capability(self):
        response=self.client.post("/dispatch",json={"capability":"code","prompt":"write a function"})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["result"]["text"],"generated code")
    def test_dispatch_unconfigured_capability_returns_503(self):
        response=self.client.post("/dispatch",json={"capability":"3d","prompt":"create a model"})
        self.assertEqual(response.status_code,503)

if __name__=="__main__": unittest.main()
