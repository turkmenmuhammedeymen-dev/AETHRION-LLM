import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from fastapi.testclient import TestClient
from aethrion import api
from aethrion.api import build_router
from aethrion.routing import Capability,CapabilityRequest,CapabilityRouter,CapabilityUnavailable

class ApiRegistrationTests(unittest.TestCase):
    def test_empty_environment_registers_no_remote_provider(self):
        env={"AETHRION_CODE_API_KEY":"","OPENAI_API_KEY":"","AETHRION_CODE_MODEL":"",
             "AETHRION_IMAGE_API_KEY":"","MESHY_API_KEY":"","TAVILY_API_KEY":"",
             "AETHRION_CHECKPOINT":"/missing/checkpoint.pt","AETHRION_TOKENIZER":"/missing/tokenizer.json"}
        with patch.dict("os.environ",env,clear=False):
            router,providers=build_router()
        self.assertEqual(router.available(), {})
        self.assertIn("runtime", providers)
        self.assertEqual(providers["runtime"].status()["chat"], "not_trained")
    def test_remote_providers_can_be_disabled_even_when_secrets_exist(self):
        env={"AETHRION_CODE_API_KEY":"dummy","AETHRION_CODE_MODEL":"dummy-model",
             "AETHRION_IMAGE_API_KEY":"dummy","OPENAI_API_KEY":"dummy",
             "MESHY_API_KEY":"dummy","TAVILY_API_KEY":"dummy",
             "AETHRION_AUDIO_API_KEY":"dummy",
             "AETHRION_CHECKPOINT":"/missing/checkpoint.pt",
             "AETHRION_TOKENIZER":"/missing/tokenizer.json"}
        with patch.dict("os.environ",env,clear=False):
            router,providers=build_router(allow_remote_providers=False)
        self.assertEqual(router.available(), {})
        self.assertIn("runtime", providers)
        self.assertEqual(providers["runtime"].status()["image"], "not_trained")

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

    def test_external_provider_registration_is_rejected(self):
        env={"AETHRION_CODE_API_KEY":"dummy","AETHRION_CODE_MODEL":"dummy-model",
             "AETHRION_CODE_BASE_URL":"https://example.test/v1","AETHRION_IMAGE_API_KEY":"",
             "OPENAI_API_KEY":"","MESHY_API_KEY":"","TAVILY_API_KEY":"dummy",
             "AETHRION_CHECKPOINT":"/missing/checkpoint.pt","AETHRION_TOKENIZER":"/missing/tokenizer.json"}
        with patch.dict("os.environ",env,clear=False):
            with self.assertRaisesRegex(ValueError, "native-only policy"):
                build_router(allow_remote_providers=True)


class _FakeSafetyRegistry:
    def __init__(self, statuses):
        self.statuses = statuses

    def status(self, model_id):
        return self.statuses.get(model_id, "not_trained")

    def resolve(self, model_id):
        return SimpleNamespace(architecture=f"native:{model_id}")


class _FakeSafetyRuntime:
    def __init__(self, statuses, decisions=None, output_text="safe generated output"):
        self.registry = _FakeSafetyRegistry(statuses)
        self.decisions = decisions or {}
        self.output_text = output_text
        self.reviewed = []
        self.research_index_valid = True
        self.audio_path = None
        self.spoken_text = "safe spoken output"

    def validate_research_index(self):
        if not self.research_index_valid:
            raise api.NativeModelError("Research index is not verified.")
        return True

    def transcribe_wav(self, path):
        return self.spoken_text

    def classify_safety(self, text):
        self.reviewed.append(text)
        return {"decision": self.decisions.get(text, "ALLOW"), "confidence": 0.99}

    def handle(self, capability, request):
        if capability == "safety":
            return {"decision": self.decisions.get(request.prompt, "ALLOW")}
        if capability == "tts":
            return {"audio_path": self.audio_path}
        return {"text": self.output_text}


class SafeRuntimeGateTests(unittest.TestCase):
    def _build(self, statuses, decisions=None, output_text="safe generated output"):
        runtime = _FakeSafetyRuntime(statuses, decisions, output_text)
        with patch.object(api, "NativeModelRuntime", return_value=runtime):
            router, providers = build_router(allow_remote_providers=False)
        return router, providers, runtime

    def test_untrained_safe_model_disables_other_ready_capabilities(self):
        router, _, _ = self._build({"chat": "ready", "safety": "not_trained"})
        self.assertNotIn("chat", router.available())
        self.assertNotIn("safety", router.available())

    def test_tts_requires_both_vocoder_and_asr(self):
        cases = [
            {"tts": "ready", "safety": "ready", "vocoder": "ready"},
            {"tts": "ready", "safety": "ready", "asr": "ready"},
        ]
        for statuses in cases:
            with self.subTest(statuses=statuses):
                router, _, _ = self._build(statuses)
                self.assertNotIn("tts", router.available())

    def test_tts_reviews_transcribed_audio_and_deletes_blocked_output(self):
        runtime = _FakeSafetyRuntime({
            "tts": "ready", "vocoder": "ready", "asr": "ready", "safety": "ready",
        }, decisions={"unsafe spoken output": "BLOCK"})
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "tts.wav"
            audio.write_bytes(b"synthetic-audio")
            runtime.audio_path = str(audio)
            runtime.spoken_text = "unsafe spoken output"
            with patch.object(api, "NativeModelRuntime", return_value=runtime):
                router, _ = build_router(allow_remote_providers=False)
            self.assertIn("tts", router.available())
            with self.assertRaises(CapabilityUnavailable):
                router.dispatch("tts", CapabilityRequest("hello", {}))
            self.assertFalse(audio.exists())

    def test_research_is_not_registered_without_verified_local_index(self):
        runtime = _FakeSafetyRuntime({"research": "ready", "safety": "ready"})
        runtime.research_index_valid = False
        with patch.object(api, "NativeModelRuntime", return_value=runtime):
            router, _ = build_router(allow_remote_providers=False)
        self.assertNotIn("research", router.available())

    def test_image_and_3d_output_models_stay_unavailable_without_visual_output_safety(self):
        router, _, _ = self._build({
            "safety": "ready", "image": "ready", "3d": "ready",
            "chat": "ready", "code": "ready",
        })
        self.assertIn("chat", router.available())
        self.assertIn("code", router.available())
        self.assertNotIn("image", router.available())
        self.assertNotIn("3d", router.available())

    def test_safe_allow_is_required_for_prompt_and_generated_text(self):
        router, _, runtime = self._build({"chat": "ready", "safety": "ready"})
        result = router.dispatch("chat", CapabilityRequest("hello", {}))
        self.assertEqual(result.output["text"], "safe generated output")
        self.assertEqual(runtime.reviewed, ["hello", "safe generated output"])

    def test_blocked_prompt_is_not_forwarded_to_chat(self):
        router, _, runtime = self._build(
            {"chat": "ready", "safety": "ready"},
            decisions={"blocked prompt": "BLOCK"},
        )
        with self.assertRaises(CapabilityUnavailable):
            router.dispatch("chat", CapabilityRequest("blocked prompt", {}))
        self.assertEqual(runtime.reviewed, ["blocked prompt"])

    def test_reviewed_generated_output_is_not_returned(self):
        router, _, runtime = self._build(
            {"chat": "ready", "safety": "ready"},
            decisions={"generated unsafe output": "REVIEW"},
            output_text="generated unsafe output",
        )
        with self.assertRaises(CapabilityUnavailable):
            router.dispatch("chat", CapabilityRequest("benign prompt", {}))
        self.assertEqual(runtime.reviewed, ["benign prompt", "generated unsafe output"])



    def test_safe_low_or_invalid_confidence_never_allows(self):
        class _LowConfidenceRuntime(_FakeSafetyRuntime):
            def classify_safety(self, text):
                self.reviewed.append(text)
                return {"decision": "ALLOW", "confidence": 0.4}
        runtime = _LowConfidenceRuntime({"safety": "ready"})
        with self.assertRaises(CapabilityUnavailable):
            api._require_safety_allow(runtime, "uncertain")
        class _InvalidConfidenceRuntime(_FakeSafetyRuntime):
            def classify_safety(self, text):
                self.reviewed.append(text)
                return {"decision": "ALLOW", "confidence": float("nan")}
        runtime = _InvalidConfidenceRuntime({"safety": "ready"})
        with self.assertRaises(CapabilityUnavailable):
            api._require_safety_allow(runtime, "invalid score")

    def test_router_screens_long_prompt_and_generated_output_in_chunks(self):
        router, _, runtime = self._build({"chat": "ready", "safety": "ready"}, output_text="O" * 1001)
        router.dispatch("chat", CapabilityRequest("P" * 1001, {}))
        self.assertEqual([len(x) for x in runtime.reviewed], [400, 400, 201, 400, 400, 201])

    def test_generic_dispatch_cannot_use_document_filesystem_paths(self):
        client=TestClient(api.app)
        with patch.dict("os.environ", {"AETHRION_API_TOKEN":"test-api-token"}):
            response=client.post(
                "/dispatch",
                headers={"Authorization":"Bearer test-api-token"},
                json={"capability":"document","prompt":"read file","options":{"document_path":"/etc/passwd"}},
            )
        self.assertEqual(response.status_code,400)

    def test_safe_chunk_gate_checks_every_segment(self):
        runtime = _FakeSafetyRuntime({"safety": "ready"})
        api._require_safety_allow_chunks(runtime, "x" * 1001, chunk_chars=400)
        self.assertEqual(len(runtime.reviewed), 3)
        self.assertEqual([len(item) for item in runtime.reviewed], [400, 400, 201])

    def test_safe_chunk_gate_stops_on_first_non_allow_segment(self):
        runtime = _FakeSafetyRuntime(
            {"safety": "ready"},
            decisions={"BLOCK" + "B" * 395: "BLOCK"},
        )
        with self.assertRaises(CapabilityUnavailable):
            api._require_safety_allow_chunks(runtime, "A" * 400 + "BLOCK" + "B" * 400, chunk_chars=400)
        self.assertEqual(len(runtime.reviewed), 2)


    def test_safe_chunk_http_wrapper_returns_service_unavailable(self):
        runtime = _FakeSafetyRuntime({"safety": "ready"}, decisions={"BLOCK": "BLOCK"})
        with self.assertRaises(api.HTTPException) as caught:
            api._require_safety_allow_chunks_http(runtime, "BLOCK")
        self.assertEqual(caught.exception.status_code, 503)


    def test_research_excerpts_are_included_in_output_safety_review(self):
        self.assertEqual(
            api._text_for_safety({"results": [{"title": "Title", "excerpt": "Passage"}]}),
            "Passage\nTitle",
        )

    def test_document_creation_fails_closed_when_safe_is_not_released(self):
        runtime = _FakeSafetyRuntime({"safety": "not_trained"})
        with patch.object(api, "providers", {"runtime": runtime}):
            with self.assertRaises(api.HTTPException) as caught:
                api.documents_create({"title": "Test", "content": "Safe sample."})
        self.assertEqual(caught.exception.status_code, 503)

    def test_document_creation_rejects_non_string_fields(self):
        runtime = _FakeSafetyRuntime({"safety": "ready"})
        with patch.object(api, "providers", {"runtime": runtime}):
            with self.assertRaises(api.HTTPException) as caught:
                api.documents_create({"title": ["not", "text"], "content": "Content"})
        self.assertEqual(caught.exception.status_code, 400)

    def test_safe_classifier_route_does_not_gate_itself(self):
        router, _, runtime = self._build(
            {"safety": "ready"},
            decisions={"classify this": "REVIEW"},
        )
        result = router.dispatch("safety", CapabilityRequest("classify this", {}))
        self.assertEqual(result.output["decision"], "REVIEW")
        self.assertEqual(runtime.reviewed, [])


class ApiHttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.router=CapabilityRouter()
        self.router.register(Capability.CODE,"test-code",lambda req:{"text":"generated code"})
        self.router.register(Capability.CHAT,"test-chat",lambda req:{"text":"chat reply"})
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

    def test_every_non_health_api_route_requires_bearer_token(self):
        from fastapi.routing import APIRoute
        from aethrion.security import require_api_token

        api_routes = [route for route in api.app.routes if isinstance(route, APIRoute)]
        self.assertGreater(len(api_routes), 1)
        unprotected = sorted(
            route.path
            for route in api_routes
            if route.path != "/health"
            and not any(
                dependency.call is require_api_token
                for dependency in route.dependant.dependencies
            )
        )
        self.assertEqual(unprotected, [])

    def test_protected_routes_fail_closed_when_server_token_is_unset(self):
        for configured_token in ("", "   "):
            with patch.dict("os.environ", {"AETHRION_API_TOKEN": configured_token}):
                response=self.client.get("/capabilities")
            self.assertEqual(response.status_code,503)

    def test_protected_routes_reject_missing_or_invalid_bearer_token(self):
        missing=self.client.get("/capabilities")
        invalid=self.client.get("/capabilities",headers={"Authorization":"Bearer wrong-token"})
        self.assertEqual(missing.status_code,401)
        self.assertEqual(invalid.status_code,401)

    def test_dispatch_rejects_server_filesystem_path_capabilities(self):
        cases = [
            ("document", {"document_path": "/etc/passwd"}),
            ("vision", {"image_path": "/etc/passwd"}),
            ("asr", {"audio_path": "/etc/passwd"}),
            ("image", {"output_path": "/etc/passwd"}),
            ("3d", {"output_path": "/etc/passwd"}),
            ("tts", {"output_path": "/etc/passwd"}),
        ]
        for capability, options in cases:
            with self.subTest(capability=capability):
                response=self.client.post(
                    "/dispatch",
                    headers=self.auth,
                    json={"capability":capability,"prompt":"inspect file","options":options},
                )
                self.assertEqual(response.status_code,400)
                self.assertIn("dedicated",response.json()["detail"])

    def test_capabilities_endpoint_reports_registered_handlers(self):
        response=self.client.get("/capabilities",headers=self.auth)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["available"],{"code":"test-code","chat":"test-chat"})
        self.assertIn("image",response.json()["unavailable"])
        self.assertEqual(response.json()["assistant_entrypoint"], "/assistant")
        self.assertEqual(response.json()["default_model"], "chat")
        self.assertEqual(response.json()["internal_dependencies"]["tts"], ["vocoder", "asr"])
    def test_dispatch_endpoint_routes_selected_capability(self):
        response=self.client.post("/dispatch",json={"capability":"code","prompt":"write a function"},headers=self.auth)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["result"]["text"],"generated code")
    def test_dispatch_rejects_file_output_capability(self):
        response=self.client.post("/dispatch",json={"capability":"3d","prompt":"create a model"},headers=self.auth)
        self.assertEqual(response.status_code,400)
        self.assertIn("dedicated",response.json()["detail"])

    def test_assistant_endpoint_delegates_clear_code_tasks(self):
        response = self.client.post(
            "/assistant", headers=self.auth,
            json={"prompt": "Write a Python function to sort records."},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["capability"], "code")
        self.assertTrue(response.json()["delegated"])
        self.assertEqual(response.json()["result"]["text"], "generated code")

    def test_assistant_endpoint_uses_chat_for_general_conversation(self):
        response = self.client.post(
            "/assistant", headers=self.auth,
            json={"prompt": "Explain why the sky looks blue."},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["capability"], "chat")
        self.assertFalse(response.json()["delegated"])
        self.assertEqual(response.json()["result"]["text"], "chat reply")

    def test_assistant_endpoint_rejects_server_paths(self):
        response = self.client.post(
            "/assistant", headers=self.auth,
            json={"prompt": "Describe this image", "options": {"image_path": "/etc/passwd"}},
        )
        self.assertEqual(response.status_code, 400)

    def test_assistant_endpoint_requires_attachment_for_vision(self):
        response = self.client.post(
            "/assistant", headers=self.auth,
            json={"prompt": "Describe this image"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("image_path", response.json()["detail"])

    def test_assistant_upload_routes_image_to_vision_with_bounded_temp_file(self):
        observed = {}
        def vision_handler(request):
            path = Path(request.options["image_path"])
            observed["bytes"] = path.read_bytes()
            observed["prompt"] = request.prompt
            observed["exists_during_dispatch"] = path.is_file()
            observed["temporary_path"] = str(path)
            return {"similarity": 0.75}
        self.router.register(Capability.VISION, "test-vision", vision_handler)
        response = self.client.post(
            "/assistant/upload",
            headers=self.auth,
            data={"prompt": "Describe this image"},
            files={"file": ("scene.png", b"PNG-test-data", "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["capability"], "vision")
        self.assertTrue(response.json()["delegated"])
        self.assertEqual(observed["bytes"], b"PNG-test-data")
        self.assertTrue(observed["exists_during_dispatch"])
        self.assertFalse(Path(observed["temporary_path"]).exists())
        self.assertEqual(observed["prompt"], "Describe this image")

    def test_assistant_upload_routes_audio_and_document_to_matching_specialists(self):
        observed = {}
        def asr_handler(request):
            path = Path(request.options["audio_path"])
            observed["audio"] = path.read_bytes()
            observed["audio_cleaned_path"] = str(path)
            return {"text": "recognized words"}
        def document_handler(request):
            path = Path(request.options["document_path"])
            observed["document"] = path.read_bytes()
            observed["document_cleaned_path"] = str(path)
            return {"text": "document summary"}
        self.router.register(Capability.ASR, "test-asr", asr_handler)
        self.router.register(Capability.DOCUMENT, "test-document", document_handler)

        audio_response = self.client.post(
            "/assistant/upload",
            headers=self.auth,
            data={"prompt": "Transcribe this audio"},
            files={"file": ("voice.wav", b"RIFF-test-audio", "audio/wav")},
        )
        document_response = self.client.post(
            "/assistant/upload",
            headers=self.auth,
            data={"prompt": "Analyze this document"},
            files={"file": ("notes.txt", b"document-body", "text/plain")},
        )
        self.assertEqual(audio_response.status_code, 200, audio_response.text)
        self.assertEqual(audio_response.json()["capability"], "asr")
        self.assertEqual(document_response.status_code, 200, document_response.text)
        self.assertEqual(document_response.json()["capability"], "document")
        self.assertEqual(observed["audio"], b"RIFF-test-audio")
        self.assertEqual(observed["document"], b"document-body")
        self.assertFalse(Path(observed["audio_cleaned_path"]).exists())
        self.assertFalse(Path(observed["document_cleaned_path"]).exists())

    def test_assistant_upload_rejects_extension_capability_mismatch(self):
        response = self.client.post(
            "/assistant/upload",
            headers=self.auth,
            data={"prompt": "Transcribe this audio"},
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_assistant_upload_rejects_non_file_intent(self):
        response = self.client.post(
            "/assistant/upload",
            headers=self.auth,
            data={"prompt": "Write a Python function"},
            files={"file": ("code.txt", b"code", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_assistant_upload_requires_authentication(self):
        response = self.client.post(
            "/assistant/upload",
            data={"prompt": "Describe this image"},
            files={"file": ("scene.png", b"PNG-test-data", "image/png")},
        )
        self.assertEqual(response.status_code, 401)

    def test_dispatch_unconfigured_text_capability_returns_503(self):
        response=self.client.post("/dispatch",json={"capability":"research","prompt":"summarize"},headers=self.auth)
        self.assertEqual(response.status_code,503)

if __name__=="__main__": unittest.main()
