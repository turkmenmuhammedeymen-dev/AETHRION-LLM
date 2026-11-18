import unittest

from aethrion.orchestrator import (
    AssistantInputRequired,
    AssistantOrchestrator,
    classify_intent,
)
from aethrion.routing import Capability, CapabilityRouter, CapabilityUnavailable


class AssistantOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.router = CapabilityRouter()
        self.calls = []
        for capability in (
            Capability.CHAT, Capability.CODE, Capability.RESEARCH,
            Capability.IMAGE, Capability.THREE_D, Capability.ASR,
            Capability.TTS, Capability.VISION, Capability.DOCUMENT,
        ):
            self.router.register(
                capability, "native-" + capability.value,
                lambda request, selected=capability: (
                    self.calls.append((selected, request.prompt, dict(request.options)))
                    or {"text": selected.value + " result"}
                ),
            )
        self.orchestrator = AssistantOrchestrator(self.router)

    def test_general_question_uses_main_chat_model(self):
        response = self.orchestrator.dispatch("Explain why the sky looks blue.")
        self.assertEqual(response.capability, Capability.CHAT)
        self.assertFalse(response.delegated)
        self.assertEqual(self.calls[0][0], Capability.CHAT)

    def test_clear_code_task_delegates_to_code_model(self):
        response = self.orchestrator.dispatch("Write a Python function to sort records.")
        self.assertEqual(response.capability, Capability.CODE)
        self.assertTrue(response.delegated)
        self.assertEqual(self.calls[0][0], Capability.CODE)

    def test_clear_research_task_delegates_to_research_model(self):
        response = self.orchestrator.dispatch("Araştır: yapay zekâ güvenliği kaynak bul.")
        self.assertEqual(response.capability, Capability.RESEARCH)

    def test_explicit_capability_overrides_heuristic(self):
        response = self.orchestrator.dispatch(
            "Hello", {"capability": "code"}
        )
        self.assertEqual(response.capability, Capability.CODE)
        self.assertNotIn("capability", self.calls[0][2])

    def test_ambiguous_multiple_tasks_fall_back_to_chat(self):
        intent = classify_intent("Write code and generate an image")
        self.assertEqual(intent.capability, Capability.CHAT)
        self.assertEqual(intent.reason, "ambiguous_multiple_task_cues")

    def test_document_vision_and_asr_require_their_input(self):
        cases = [
            ("Analyze this document", "document_path"),
            ("Describe this image", "image_path"),
            ("Transcribe audio", "audio_path"),
        ]
        for prompt, required in cases:
            with self.subTest(prompt=prompt):
                with self.assertRaises(AssistantInputRequired) as error:
                    self.orchestrator.dispatch(prompt)
                self.assertIn(required, str(error.exception))

    def test_each_specialist_task_has_a_deterministic_route(self):
        cases = [
            ("Create an image of a mountain", Capability.IMAGE, {}),
            ("Create a 3D model of a chair", Capability.THREE_D, {}),
            ("Read this aloud", Capability.TTS, {}),
            ("Transcribe audio", Capability.ASR, {"audio_path": "/tmp/voice.wav"}),
            ("Analyze this document", Capability.DOCUMENT, {"document_path": "/tmp/file.pdf"}),
            ("Describe this image", Capability.VISION, {"image_path": "/tmp/photo.png"}),
            ("Research this topic and find sources", Capability.RESEARCH, {}),
            ("Write code to sort records", Capability.CODE, {}),
        ]
        for prompt, expected, options in cases:
            with self.subTest(prompt=prompt):
                intent = classify_intent(prompt)
                self.assertEqual(intent.capability, expected)
                if options:
                    response = self.orchestrator.dispatch(prompt, options)
                    self.assertEqual(response.capability, expected)

    def test_attachment_is_forwarded_to_matching_specialist(self):
        response = self.orchestrator.dispatch(
            "Describe this image", {"image_path": "/tmp/input.png"}
        )
        self.assertEqual(response.capability, Capability.VISION)
        self.assertEqual(self.calls[0][2]["image_path"], "/tmp/input.png")

    def test_safe_is_not_a_user_task_route(self):
        with self.assertRaisesRegex(ValueError, "policy gate"):
            self.orchestrator.dispatch("hello", {"capability": "safety"})

    def test_unavailable_specialist_does_not_silently_fall_back_to_chat(self):
        router = CapabilityRouter()
        router.register(Capability.CHAT, "chat", lambda request: "chat")
        with self.assertRaises(CapabilityUnavailable):
            AssistantOrchestrator(router).dispatch("Write code for me")


if __name__ == "__main__":
    unittest.main()
