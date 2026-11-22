"""Ask the running Space to run bounded training sessions (CircleCI, approval-gated).
Needs: HF_TOKEN (CircleCI context) and the Space login AETHRION_SPACE_USERNAME / AETHRION_SPACE_PASSWORD in the same context.
Usage: python ci/trigger_space_training.py [--models code research ...] [--steps 25] [--rounds 1]"""
from __future__ import annotations
import argparse
import os
import sys

SPACE_ID = "vizyon2027/AETHRION-LLM"
DEFAULT_MODELS = ["code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document"]


def run(client, models, steps, rounds):
    summary = []
    for _ in range(rounds):
        for model in models:
            out = client.predict(model, steps, api_name="/training_session")
            ok = "Training session finished" in str(out) and "Trainer ok: True" in str(out)
            print(f"[{model}] {'OK' if ok else 'NOT OK'}\n{str(out)[:600]}\n", flush=True)
            summary.append((model, ok))
    return summary


def _credential_candidates(user, pw):
    """The Space compares the password exactly (the username is stripped there), so try raw and trimmed values."""
    out = []
    for cand in ((user.strip(), pw), (user.strip(), pw.strip()), (user.strip(), pw.rstrip("\r\n"))):
        if cand not in out:
            out.append(cand)
    return out


def _probe_login(space_url, token, user, pw):
    try:
        import requests
        r = requests.post(space_url.rstrip("/") + "/login", data={"username": user, "password": pw},
                          headers={"Authorization": f"Bearer {token}"}, timeout=30)
        return f"HTTP {r.status_code} {r.text[:120]!r}"
    except Exception as exc:  # diagnostics only
        return f"probe failed: {type(exc).__name__}"


def make_client(space_id, token, auth, client_cls=None):
    """gradio_client renamed `hf_token` to `token` in newer releases; support both. Retries credential variants."""
    import inspect
    if client_cls is None:
        from gradio_client import Client as client_cls
    params = inspect.signature(client_cls.__init__).parameters
    key = "token" if "token" in params else "hf_token"
    last = None
    for user, pw in _credential_candidates(*auth):
        try:
            return client_cls(space_id, auth=(user, pw), **{key: token})
        except ValueError as exc:  # "Could not login"
            last = exc
    user, pw = auth
    url = "https://" + space_id.replace("/", "-").replace("_", "-").lower() + ".hf.space"
    raise RuntimeError(
        "Space login rejected. Check that AETHRION_SPACE_USERNAME / AETHRION_SPACE_PASSWORD in the CircleCI context are "
        f"EXACTLY the Space secrets of the same names. (username length {len(user)}, password length {len(pw)}; "
        f"server answer: {_probe_login(url, token, user.strip(), pw)})"
    ) from last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    token, user, pw = os.environ.get("HF_TOKEN"), os.environ.get("AETHRION_SPACE_USERNAME"), os.environ.get("AETHRION_SPACE_PASSWORD")
    if not (token and user and pw):
        print("Missing HF_TOKEN / AETHRION_SPACE_USERNAME / AETHRION_SPACE_PASSWORD in the CircleCI context 'aethrion-deploy'.", file=sys.stderr)
        return 2
    client = make_client(SPACE_ID, token, (user, pw))
    results = run(client, a.models, a.steps, a.rounds)
    failed = [m for m, ok in results if not ok]
    print("SUMMARY", {"ok": [m for m, ok in results if ok], "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
