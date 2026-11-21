import unittest
from aethrion import openai_compat as oc


class OpenAICompatTests(unittest.TestCase):
    def test_untrained_models_are_not_reported_ready(self):
        out = oc.list_models({}, {"chat": "not_trained"})
        self.assertTrue(all(not m["available"] for m in out["data"]))
        self.assertEqual({m["status"] for m in out["data"]}, {"not_trained"})

    def test_registered_capability_is_ready(self):
        out = oc.list_models({"chat": "aethrion.model:AETHRIONLM"})
        chat = next(m for m in out["data"] if m["id"] == "aethrion-chat")
        self.assertTrue(chat["available"]); self.assertEqual(chat["status"], "ready")

    def test_parse_valid_request(self):
        cap, prompt, opts = oc.parse_chat_request({
            "model": "aethrion-chat", "max_tokens": 64, "temperature": 0.5,
            "messages": [{"role": "user", "content": "Merhaba AETHRION"}]})
        self.assertEqual(cap, "chat"); self.assertEqual(prompt, "user: Merhaba AETHRION")
        self.assertEqual(opts, {"max_new_tokens": 64, "temperature": 0.5})

    def test_rejections(self):
        bad = [
            None, {}, {"model": "gpt-4", "messages": []},
            {"model": "aethrion-image", "messages": [{"role": "user", "content": "x"}]},
            {"model": "aethrion-chat", "messages": []},
            {"model": "aethrion-chat", "messages": [{"role": "tool", "content": "x"}]},
            {"model": "aethrion-chat", "messages": [{"role": "user", "content": "  "}]},
            {"model": "aethrion-chat", "stream": True, "messages": [{"role": "user", "content": "x"}]},
            {"model": "aethrion-chat", "max_tokens": True, "messages": [{"role": "user", "content": "x"}]},
            {"model": "aethrion-chat", "max_tokens": 99999, "messages": [{"role": "user", "content": "x"}]},
            {"model": "aethrion-chat", "temperature": 9, "messages": [{"role": "user", "content": "x"}]},
            {"model": "aethrion-chat", "messages": [{"role": "user", "content": "a" * 13000}]},
        ]
        for payload in bad:
            with self.assertRaises(oc.ContractError, msg=str(payload)[:60]):
                oc.parse_chat_request(payload)

    def test_response_shape_and_bad_output(self):
        r = oc.build_chat_response("aethrion-chat", {"text": "Merhaba"}, "user: hi")
        self.assertEqual(r["choices"][0]["message"]["content"], "Merhaba")
        self.assertTrue(r["usage"]["estimated"])
        for bad in ({"x": 1}, "", "  "):
            with self.assertRaises(oc.ContractError):
                oc.build_chat_response("aethrion-chat", bad, "p")


if __name__ == "__main__":
    unittest.main()
