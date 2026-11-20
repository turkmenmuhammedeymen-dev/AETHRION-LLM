from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace

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
