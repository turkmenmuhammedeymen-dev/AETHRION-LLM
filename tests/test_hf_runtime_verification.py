from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError

from ci.hf_runtime import (
    SpaceRuntimeVerificationError,
    probe_gradio_config,
    wait_for_space_running,
)


class RuntimeVerificationTests(unittest.TestCase):
    def test_waits_until_space_is_running(self):
        class API:
            def __init__(self):
                self.stages = iter(["BUILDING", "RUNNING"])

            def get_space_runtime(self, repo_id):
                return SimpleNamespace(stage=next(self.stages))

        stage = wait_for_space_running(
            API(), "vizyon2027/AETHRION-LLM",
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
        )
        self.assertEqual(stage, "RUNNING")

    def test_fails_on_build_error(self):
        class API:
            def get_space_runtime(self, repo_id):
                return SimpleNamespace(stage="BUILD_ERROR")

        with self.assertRaisesRegex(SpaceRuntimeVerificationError, "BUILD_ERROR"):
            wait_for_space_running(API(), "vizyon2027/AETHRION-LLM")

    def test_does_not_restart_paused_space(self):
        class API:
            def get_space_runtime(self, repo_id):
                return SimpleNamespace(stage="PAUSED")

        with self.assertRaisesRegex(SpaceRuntimeVerificationError, "refusing to change"):
            wait_for_space_running(API(), "vizyon2027/AETHRION-LLM")

    def test_gradio_config_endpoint_is_validated(self):
        payload = {"components": [{"id": 1}], "dependencies": []}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(payload).encode("utf-8")

        def open_url(_request, timeout):
            self.assertEqual(timeout, 5)
            return Response()

        config = probe_gradio_config(
            "vizyon2027/AETHRION-LLM", timeout_seconds=5, open_url=open_url
        )
        self.assertEqual(config["components"], payload["components"])

    def test_falls_back_to_root_html_when_config_endpoint_is_404(self):
        payload = {"components": [{"id": 7}], "dependencies": [{"id": 1}]}
        html = (
            "<html><script>window.gradio_config = "
            + json.dumps(payload)
            + ";</script></html>"
        )
        calls = []

        class Response:
            status = 200

            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return self.body

        def open_url(request, **_kwargs):
            calls.append(request.full_url)
            if request.full_url.endswith("/config"):
                raise HTTPError(request.full_url, 404, "Not Found", None, None)
            return Response(html.encode("utf-8"))

        config = probe_gradio_config(
            "vizyon2027/AETHRION-LLM", open_url=open_url
        )
        self.assertEqual(config, payload)
        self.assertEqual(
            calls,
            [
                "https://vizyon2027-aethrion-llm.hf.space/config",
                "https://vizyon2027-aethrion-llm.hf.space/",
            ],
        )

    def test_reports_root_404_after_config_404(self):
        def open_url(request, **_kwargs):
            raise HTTPError(request.full_url, 404, "Not Found", None, None)

        with self.assertRaisesRegex(
            SpaceRuntimeVerificationError, "root endpoint returned HTTP 404"
        ):
            probe_gradio_config(
                "vizyon2027/AETHRION-LLM", open_url=open_url
            )

    def test_accepts_auth_challenge_as_live_protected_app(self):
        def open_url(_request, **_kwargs):
            raise HTTPError("https://example.hf.space/config", 401, "Unauthorized", None, None)

        result = probe_gradio_config(
            "vizyon2027/AETHRION-LLM", open_url=open_url
        )
        self.assertEqual(result, {"auth_required": True})

    def test_rejects_non_gradio_response(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return io.BytesIO(b'{"ok": true}').read()

        with self.assertRaisesRegex(SpaceRuntimeVerificationError, "components"):
            probe_gradio_config(
                "vizyon2027/AETHRION-LLM",
                open_url=lambda *_args, **_kwargs: Response(),
            )


if __name__ == "__main__":
    unittest.main()
