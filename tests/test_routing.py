import unittest

from aethrion.routing import (
    Capability,
    CapabilityRequest,
    CapabilityRouter,
    CapabilityUnavailable,
)


class CapabilityRouterTests(unittest.TestCase):
    def test_selected_capability_uses_only_its_registered_handler(self):
        router = CapabilityRouter()
        calls = []
        router.register(Capability.CHAT, "aethrion-text", lambda req: calls.append("chat") or "text")
        router.register(Capability.CODE, "code-agent", lambda req: calls.append("code") or "patch")

        result = router.dispatch(Capability.CODE, CapabilityRequest("Write a Python function"))

        self.assertEqual(result.capability, Capability.CODE)
        self.assertEqual(result.provider, "code-agent")
        self.assertEqual(result.output, "patch")
        self.assertEqual(calls, ["code"])

    def test_unconfigured_image_or_3d_provider_fails_clearly(self):
        router = CapabilityRouter()
        for capability in (Capability.IMAGE, Capability.THREE_D):
            with self.subTest(capability=capability):
                with self.assertRaises(CapabilityUnavailable):
                    router.dispatch(capability, CapabilityRequest("Create an asset"))

    def test_duplicate_registration_requires_explicit_replace(self):
        router = CapabilityRouter()
        router.register("chat", "first", lambda req: "one")
        with self.assertRaises(ValueError):
            router.register("chat", "second", lambda req: "two")
        router.register("chat", "second", lambda req: "two", replace=True)
        self.assertEqual(router.available()["chat"], "second")

    def test_empty_prompt_is_rejected(self):
        router = CapabilityRouter()
        router.register("chat", "text", lambda req: "ok")
        with self.assertRaises(ValueError):
            router.dispatch("chat", CapabilityRequest("   "))

    def test_provider_must_return_a_result(self):
        router = CapabilityRouter()
        router.register("image", "image-provider", lambda req: None)
        with self.assertRaisesRegex(RuntimeError, "returned no result"):
            router.dispatch("image", CapabilityRequest("A mountain"))

    def test_available_does_not_expose_secrets(self):
        router = CapabilityRouter()
        router.register("3d", "mesh-provider", lambda req: {"asset": "model.glb"})
        self.assertEqual(router.available(), {"3d": "mesh-provider"})


if __name__ == "__main__":
    unittest.main()
