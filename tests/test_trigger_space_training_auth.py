"""Safe, offline tests for the GitHub Actions -> Gradio Space authentication bridge."""
from __future__ import annotations

import pytest

from ci.trigger_space_training import (
    SpaceAuthenticationError,
    login_cookies,
    make_client,
)


class FakeResponse:
    def __init__(self, status_code, cookies=None, text=""):
        self.status_code = status_code
        self.cookies = cookies or {}
        self.text = text


def test_login_sends_hf_token_only_as_authorization_header_for_private_space():
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(200, {"access-token-test": "session-cookie"})

    cookies = login_cookies(
        "https://example.hf.space/",
        " owner ",
        " exact password ",
        "hf-test-token",
        post=fake_post,
    )

    assert cookies == {"access-token-test": "session-cookie"}
    assert calls == [
        (
            "https://example.hf.space/login",
            {
                "data": {"username": " owner ", "password": " exact password "},
                "timeout": 30,
                "headers": {"Authorization": "Bearer hf-test-token"},
            },
        )
    ]
    assert "hf-test-token" not in str(calls[0][1]["data"])


def test_login_400_is_classified_without_echoing_response_or_credentials():
    username = "private-user"
    password = "never-print-this"
    response_body = f"Incorrect credentials: {username} {password}"

    def fake_post(url, **kwargs):
        return FakeResponse(400, text=response_body)

    with pytest.raises(SpaceAuthenticationError) as exc_info:
        login_cookies(
            "https://example.hf.space",
            username,
            password,
            post=fake_post,
        )

    message = str(exc_info.value)
    assert "HTTP 400" in message
    assert username not in message
    assert password not in message
    assert response_body not in message


def test_make_client_passes_exact_password_and_normalized_username():
    captured = {}

    class TokenClient:
        def __init__(self, src, auth=None, token=None):
            captured.update(src=src, auth=auth, token=token)

    client = make_client(
        "owner/private-space",
        "hf-test-token",
        (" owner ", " password with intentional spaces "),
        client_cls=TokenClient,
    )

    assert isinstance(client, TokenClient)
    assert captured == {
        "src": "owner/private-space",
        "auth": ("owner", " password with intentional spaces "),
        "token": "hf-test-token",
    }


def test_make_client_supports_legacy_hf_token_keyword():
    captured = {}

    class LegacyClient:
        def __init__(self, src, auth=None, hf_token=None):
            captured.update(src=src, auth=auth, hf_token=hf_token)

    make_client(
        "owner/private-space",
        "hf-test-token",
        ("owner", "password"),
        client_cls=LegacyClient,
    )

    assert captured == {
        "src": "owner/private-space",
        "auth": ("owner", "password"),
        "hf_token": "hf-test-token",
    }


def test_parse_session_result_accepts_successful_paused_session():
    from ci.trigger_space_training import parse_session_result

    output = """### Training session paused / incomplete (not released)
- Model: code | steps requested: 25
- Trainer ok: True | pulled: 4 files | pushed: 4 files
- Progress: training_interrupted_not_released step 25 | complete: False
- Released: NO — evaluation and human approval are still required.
"""
    parsed = parse_session_result(output)
    assert parsed == {
        "trainer_ok": True,
        "progress": {
            "status": "training_interrupted_not_released",
            "step": 25,
            "complete": False,
        },
        "complete": False,
    }


def test_parse_session_result_accepts_completed_session():
    from ci.trigger_space_training import parse_session_result

    output = """### Training session completed (not released)
- Trainer ok: True | pulled: 4 files | pushed: 4 files
- Progress: training_complete_not_released step 500 | complete: True
"""
    parsed = parse_session_result(output)
    assert parsed["trainer_ok"] is True
    assert parsed["complete"] is True
    assert parsed["progress"]["step"] == 500


def test_parse_session_result_rejects_failed_or_unrecognized_session():
    from ci.trigger_space_training import parse_session_result

    failed = "### Training session failed\n- Trainer ok: False | pulled: 4 files | pushed: 0 files"
    assert parse_session_result(failed)["trainer_ok"] is False
    assert parse_session_result("Training session finished successfully")["trainer_ok"] is False


def test_tail_returns_the_end_of_long_diagnostics():
    from ci.trigger_space_training import _tail

    value = "start " + ("x" * 100) + " FINAL_TRACEBACK"
    tail = _tail(value, 20)
    assert len(tail) == 20
    assert tail.endswith(" FINAL_TRACEBACK")
