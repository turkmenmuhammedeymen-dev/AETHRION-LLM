import unittest
import tempfile
from pathlib import Path
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
    def test_remote_providers_can_be_disabled_even_when_secrets_exist(self):
        env={"AETHRION_CODE_API_KEY":"dummy","AETHRION_CODE_MODEL":"dummy-model",
             "AETHRION_IMAGE_API_KEY":"dummy","OPENAI_API_KEY":"dummy",
             "MESHY_API_KEY":"dummy","TAVILY_API_KEY":"dummy",
             "AETHRION_AUDIO_API_KEY":"dummy",
             "AETHRION_CHECKPOINT":"/missing/checkpoint.pt",
             "AETHRION_TOKENIZER":"/missing/tokenizer.json"}
        with patch.dict("os.environ",env,clear=False):
            router,providers=build_router(allow_remote_providers=False)
        self.assertEqual(router.available(),{})
        self.assertEqual(providers,{})

    def test_local_model_can_be_disabled_without_authentication(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "checkpoint.pt"
            tokenizer = Path(folder) / "tokenizer.json"
            checkpoint.touch()
            tokenizer.touch()
            env = {
                "AETHRION_CHECKPOINT": str(checkpoint),
                "AETHRION_TOKENIZER": str(tokenizer),
                "AETHRION_CODE_API_KEY": "",
                "AETHRION_CODE_MODEL": "",
                "OPENAI_API_KEY": "",
                "AETHRION_IMAGE_API_KEY": "",
                "MESHY_API_KEY": "",
                "TAVILY_API_KEY": "",
                "AETHRION_AUDIO_API_KEY": "",
            }
            with patch.dict("os.environ", env, clear=False):
                router, providers = build_router(
                    allow_remote_providers=False,
                    allow_local_model=False,
                )
            self.assertEqual(router.available(), {})
            self.assertEqual(providers, {})

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
        self.env_patch=patch.dict("os.environ", {"AETHRION_API_TOKEN": "test-api-token"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.auth={"Authorization": "Bearer test-api-token"}
        self.patch=patch.object(api,"router",self.router)
        self.patch.start()
        self.addCleanup(self.patch.stop)
    def test_health_endpoint_remains_public(self):
        response=self.client.get("/health")
        self.assertEqual(response.status_code,200)

    def test_protected_routes_reject_missing_or_invalid_bearer_token(self):
        missing=self.client.get("/capabilities")
        invalid=self.client.get("/capabilities",headers={"Authorization":"Bearer wrong-token"})
        self.assertEqual(missing.status_code,401)
        self.assertEqual(invalid.status_code,401)

    def test_capabilities_endpoint_reports_registered_handlers(self):
        response=self.client.get("/capabilities",headers=self.auth)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["available"],{"code":"test-code"})
        self.assertIn("image",response.json()["unavailable"])
    def test_dispatch_endpoint_routes_selected_capability(self):
        response=self.client.post("/dispatch",json={"capability":"code","prompt":"write a function"},headers=self.auth)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["result"]["text"],"generated code")
    def test_dispatch_unconfigured_capability_returns_503(self):
        response=self.client.post("/dispatch",json={"capability":"3d","prompt":"create a model"},headers=self.auth)
        self.assertEqual(response.status_code,503)

if __name__=="__main__": unittest.main()
