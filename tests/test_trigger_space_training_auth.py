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


def test_login_posts_gradio_form_without_forwarding_hf_token():
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(200, {"access-token-test": "session-cookie"})

    cookies = login_cookies(
        "https://example.hf.space/",
        " owner ",
        " exact password ",
        post=fake_post,
    )

    assert cookies == {"access-token-test": "session-cookie"}
    assert calls == [
        (
            "https://example.hf.space/login",
            {
                "data": {"username": " owner ", "password": " exact password "},
                "timeout": 30,
            },
        )
    ]


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



def test_make_client_reports_only_safe_http_status_on_login_failure():
    username = "private-user"
    password = "never-print-this"
    response_body = f"Incorrect credentials: {username} {password}"

    class FailingClient:
        def __init__(self, src, auth=None, token=None):
            from ci.trigger_space_training import login_cookies
            def fake_post(url, **kwargs):
                return FakeResponse(401, text=response_body)
            login_cookies(src, auth[0], auth[1], post=fake_post)

    with pytest.raises(RuntimeError) as exc_info:
        make_client(
            "owner/private-space",
            "hf-test-token",
            (username, password),
            client_cls=FailingClient,
        )

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert username not in message
    assert password not in message
    assert response_body not in message
