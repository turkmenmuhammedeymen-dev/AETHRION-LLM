import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ci.provision_space_training import provision  # noqa: E402
from ci.trigger_space_training import login_cookies, make_client, run  # noqa: E402


class FakeApi:
    def __init__(self, private=True, secrets=(), variables=None):
        self.private, self.secrets, self.variables = private, list(secrets), dict(variables or {})
        self.calls = []

    def whoami(self): return {"name": "owner"}
    def create_repo(self, repo_id, **kw): self.calls.append(("create_repo", repo_id, kw["private"]))
    def repo_info(self, repo_id, repo_type): return SimpleNamespace(private=self.private)
    def get_space_variables(self, space): return self.variables
    def add_space_variable(self, space, key, value): self.calls.append(("var", key, value))
    def get_space_runtime(self, space): return SimpleNamespace(secrets=self.secrets)
    def add_space_secret(self, space, key, value, description=None): self.calls.append(("secret", key))


class ProvisionTests(unittest.TestCase):
    def test_creates_private_repo_and_sets_settings(self):
        api = FakeApi()
        out = provision(api, "tok", space_id="o/s")
        self.assertEqual(out["artifact_repo"], "owner/aethrion-artifacts")
        self.assertIn(("create_repo", "owner/aethrion-artifacts", True), api.calls)
        self.assertNotIn(("secret", "HF_TOKEN"), api.calls)  # tokens are never copied by CI
        self.assertEqual(out["hf_token_secret"], "missing")

    def test_never_overwrites_existing_secret_or_same_variable(self):
        api = FakeApi(secrets=["HF_TOKEN"], variables={"AETHRION_ARTIFACT_REPO": "owner/aethrion-artifacts"})
        out = provision(api, "tok", space_id="o/s")
        self.assertEqual(out["hf_token_secret"], "existing")
        self.assertFalse([c for c in api.calls if c[0] in ("secret", "var")])

    def test_refuses_public_repo(self):
        with self.assertRaises(RuntimeError):
            provision(FakeApi(private=False), "tok", space_id="o/s")

    def test_share_token_can_be_disabled(self):
        api = FakeApi()
        self.assertEqual(provision(api, "tok", space_id="o/s", share_token=False)["hf_token_secret"], "missing")
        self.assertNotIn(("secret", "HF_TOKEN"), api.calls)


class TriggerTests(unittest.TestCase):
    def test_run_reports_ok_and_failure(self):
        class C:
            def predict(self, model, steps, api_name):
                return "### Training session finished\n- Trainer ok: True" if model == "code" else "### Training session failed"
        res = run(C(), ["code", "safety"], 5, 1)
        self.assertEqual(res, [("code", True), ("safety", False)])


class ClientCompatTests(unittest.TestCase):
    def test_uses_token_keyword_when_available(self):
        class New:
            def __init__(self, src, token=None, auth=None): self.kw = ("token", token, auth)
        class Old:
            def __init__(self, src, hf_token=None, auth=None): self.kw = ("hf_token", hf_token, auth)
        self.assertEqual(make_client("s", "t", ("u", "p"), New).kw, ("token", "t", ("u", "p")))
        self.assertEqual(make_client("s", "t", ("u", "p"), Old).kw, ("hf_token", "t", ("u", "p")))


class LoginRetryTests(unittest.TestCase):
    def test_preserves_password_exactly_and_normalizes_only_username(self):
        class C:
            def __init__(self, src, token=None, auth=None):
                if auth != ("u", "pw\n"):
                    raise ValueError("Could not login")
        self.assertIsInstance(make_client("o/s", "t", ("u ", "pw\n"), C), C)

    def test_reports_readable_error_when_all_fail(self):
        class C:
            def __init__(self, src, token=None, auth=None): raise ValueError("Could not login")
        with self.assertRaises(RuntimeError) as ctx:
            make_client("o/s", "t", ("u", "pw"), C)
        self.assertIn("Space login rejected", str(ctx.exception))


class LoginCookieTests(unittest.TestCase):
    def test_login_sends_bearer_header_for_private_space_and_returns_cookies(self):
        seen = {}
        def fake_post(url, **kwargs):
            seen.update(url=url, **kwargs)
            return SimpleNamespace(status_code=200, cookies={"access-token": "abc"})
        out = login_cookies("https://x.hf.space/", "u", "p", "hf_tok", post=fake_post)
        self.assertEqual(out, {"access-token": "abc"})
        self.assertEqual(seen["url"], "https://x.hf.space/login")
        self.assertEqual(seen["data"], {"username": "u", "password": "p"})
        self.assertEqual(seen["headers"], {"Authorization": "Bearer hf_tok"})  # private Space: proxy 404s without it
        self.assertNotIn("hf_tok", str(seen["data"]))

    def test_retries_trailing_slash_only_after_404(self):
        seen = []
        responses = iter([
            SimpleNamespace(status_code=404, cookies={}),
            SimpleNamespace(status_code=200, cookies={"access-token": "ok"}),
        ])

        def fake_post(url, **kwargs):
            seen.append(url)
            return next(responses)

        out = login_cookies("https://x.hf.space/", "u", "p", post=fake_post)
        self.assertEqual(out, {"access-token": "ok"})
        self.assertEqual(
            seen,
            ["https://x.hf.space/login", "https://x.hf.space/login/"],
        )

    def test_login_failure_raises_sanitized_authentication_error(self):
        from ci.trigger_space_training import SpaceAuthenticationError
        statuses = iter([404, 404])
        with self.assertRaises(SpaceAuthenticationError) as ctx:
            login_cookies(
                "https://x.hf.space", "u", "p",
                post=lambda *a, **k: SimpleNamespace(
                    status_code=next(statuses), cookies={}
                ),
            )
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()


def test_run_until_complete_loops_and_stops():
    from ci.trigger_space_training import run_until_complete, parse_progress
    outs = iter([
        "### Training session finished\n- Trainer ok: True | x\n- Progress: training_interrupted_not_released step 200 | complete: False\n",
        "### Training session finished\n- Trainer ok: True | x\n- Progress: training_complete_not_released step 2000 | complete: True\n",
    ])
    class C:
        def predict(self, *a, **k): return next(outs)
    res = run_until_complete(C(), ["code"], 200, 5)
    assert res["code"]["complete"] and res["code"]["rounds"] == 2 and res["code"]["step"] == 2000
    assert parse_progress("nothing") is None
