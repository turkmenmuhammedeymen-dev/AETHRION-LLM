"""Ask the running Space to run bounded training sessions (GitHub Actions, approval-gated).
Needs: HF_TOKEN, AETHRION_SPACE_USERNAME and AETHRION_SPACE_PASSWORD from GitHub Actions repository/environment secrets.
Usage: python ci/trigger_space_training.py [--models code research ...] [--steps 25] [--rounds 1]"""
from __future__ import annotations
import argparse
import re
import os
import sys

SPACE_ID = "vizyon2027/AETHRION-LLM"
DEFAULT_MODELS = ["chat", "code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document"]


def run(client, models, steps, rounds):
    summary = []
    for _ in range(rounds):
        for model in models:
            out = client.predict(model, steps, api_name="/training_session")
            ok = "Training session finished" in str(out) and "Trainer ok: True" in str(out)
            print(f"[{model}] {'OK' if ok else 'NOT OK'}\n{str(out)[:600]}\n", flush=True)
            summary.append((model, ok))
    return summary


_PROGRESS = re.compile(r"Progress: (\S+) step (\d+|None) \| complete: (True|False)")


def parse_progress(out):
    m = _PROGRESS.search(str(out))
    if not m:
        return None
    return {"status": m.group(1), "step": None if m.group(2) == "None" else int(m.group(2)),
            "complete": m.group(3) == "True"}


def _tail(out, n=500):
    text = " ".join(str(out).split())
    return text[:n]


def run_until_complete(client, models, steps, max_rounds, min_steps=25, max_consecutive_failures=3):
    """Resumable sessions per model until training_complete_not_released or max_rounds.
    Steps are halved after a failed session (shorter sessions fit the GPU window).
    Completion never means release."""
    results = {}
    for model in models:
        cur, state = steps, {"ok": False, "complete": False, "step": None, "rounds": 0}
        failures = 0
        for _ in range(max_rounds):
            state["rounds"] += 1
            try:
                out = client.predict(model, cur, api_name="/training_session")
            except Exception as exc:  # network/quota hiccup: retry smaller
                print(f"[{model}] call failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                failures += 1
                cur = max(min_steps, cur // 2)
                if failures >= max_consecutive_failures:
                    print(f"[{model}] stopping after {failures} consecutive failed sessions", flush=True)
                    break
                continue
            ok = "Training session finished" in str(out) and "Trainer ok: True" in str(out)
            prog = parse_progress(out) or {}
            print(f"[{model}] round {state['rounds']} ok={ok} progress={prog}", flush=True)
            if not ok:
                failures += 1
                print(f"[{model}] session output (no credentials): {_tail(out)}", flush=True)
                cur = max(min_steps, cur // 2)
                if failures >= max_consecutive_failures:
                    print(f"[{model}] stopping after {failures} consecutive failed sessions", flush=True)
                    break
                continue
            failures = 0
            state.update(ok=True, step=prog.get("step"), complete=bool(prog.get("complete")))
            if state["complete"]:
                break
        results[model] = state
    return results


class SpaceAuthenticationError(RuntimeError):
    """A Gradio login failure, reported without response bodies or credentials."""

    def __init__(self, status_code: int):
        self.status_code = int(status_code)
        super().__init__(f"Gradio login returned HTTP {self.status_code}.")


def login_cookies(src, user, pw, token=None, post=None):
    """Authenticate against Gradio's form-based /login endpoint and return session cookies.

    The Space is PRIVATE: Hugging Face's front proxy answers 404 to requests without
    the owner's HF token, so the token is sent ONLY as an Authorization header to the
    Space's own host (never in the form body, never logged). The Gradio login pair is
    the form data.
    """
    if post is None and callable(token):
        # Also tolerate the former fourth-positional-argument test/client hook.
        post, token = token, None
    if post is None:
        import httpx
        post = httpx.post
    # Gradio exposes both /login and /login/. Some Space front-proxy
    # configurations resolve the slash form differently. Retry only a 404;
    # never retry rejected credentials or redirect-based auth diagnostics.
    login_base = src.rstrip("/")
    extra = {"headers": {"Authorization": f"Bearer {token}"}} if isinstance(token, str) and token else {}
    resp = post(login_base + "/login", data={"username": user, "password": pw}, timeout=30, **extra)
    if resp.status_code == 404:
        resp = post(login_base + "/login/", data={"username": user, "password": pw}, timeout=30, **extra)
    if resp.status_code != 200:
        raise SpaceAuthenticationError(resp.status_code)
    return {name: value for name, value in resp.cookies.items() if value is not None}


def make_client(space_id, token, auth, client_cls=None):
    """Build a gradio_client Client for a private Space protected by Gradio auth.

    Supports gradio_client releases exposing either the `token` or legacy
    `hf_token` constructor keyword. Credentials are passed unchanged except
    for the username whitespace normalization performed by the Space itself.
    """
    import inspect
    if client_cls is None:
        from gradio_client import Client as _Base

        class client_cls(_Base):  # noqa: N801 - local subclass
            def _login(self, auth):
                self.cookies = login_cookies(self.src, auth[0], auth[1], token)
    params = inspect.signature(client_cls.__init__).parameters
    key = "token" if "token" in params else "hf_token"
    try:
        return client_cls(space_id, auth=(auth[0].strip(), auth[1]), **{key: token})
    except SpaceAuthenticationError as exc:
        # Gradio's /login route returns 302 when auth is not configured and
        # 400 when auth is configured but the submitted pair is rejected.
        # Report only the status code; never log response bodies or secrets.
        if exc.status_code == 302:
            detail = (
                "The running Space has no Gradio auth configured. Confirm both "
                "AETHRION_SPACE_USERNAME and AETHRION_SPACE_PASSWORD exist as "
                "Space Secrets, then restart the Space so app.py reloads them."
            )
        elif exc.status_code == 400:
            detail = (
                "Gradio auth is active but rejected the submitted credentials. "
                "The GitHub Actions pair does not match the pair loaded by the "
                "running Space, or the Space is still running with old secrets."
            )
        else:
            detail = (
                f"Gradio login returned HTTP {exc.status_code}. Check Space "
                "runtime health and authentication configuration."
            )
        raise RuntimeError(
            "Space login failed (safe diagnostic): " + detail +
            " No credentials or response bodies were logged."
        ) from None
    except ValueError:
        # Other gradio_client ValueErrors may contain server details.
        raise RuntimeError(
            "Space login rejected: Gradio client initialization failed before "
            "authentication could be confirmed. No response details or "
            "credentials were logged."
        ) from None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--until-complete", action="store_true")
    a = ap.parse_args()
    token, user, pw = os.environ.get("HF_TOKEN"), os.environ.get("AETHRION_SPACE_USERNAME"), os.environ.get("AETHRION_SPACE_PASSWORD")
    if not (token and user and pw):
        print("Missing HF_TOKEN / AETHRION_SPACE_USERNAME / AETHRION_SPACE_PASSWORD in GitHub Actions secrets.", file=sys.stderr)
        return 2
    client = make_client(SPACE_ID, token, (user, pw))
    if a.until_complete:
        res = run_until_complete(client, a.models, a.steps, a.rounds)
        print("SUMMARY", res)
        return 0 if all(v["ok"] for v in res.values()) else 1
    results = run(client, a.models, a.steps, a.rounds)
    failed = [m for m, ok in results if not ok]
    print("SUMMARY", {"ok": [m for m, ok in results if ok], "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
