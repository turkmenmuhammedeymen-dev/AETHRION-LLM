import unittest
from unittest.mock import patch
from aethrion.api import build_router
from aethrion.routing import Capability

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

if __name__=="__main__": unittest.main()
