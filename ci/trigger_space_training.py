"""Ask the running Space to run bounded training sessions (GitHub Actions, approval-gated).
Needs: HF_TOKEN, AETHRION_SPACE_USERNAME and AETHRION_SPACE_PASSWORD from GitHub Actions repository/environment secrets.
Usage: python ci/trigger_space_training.py [--models code research ...] [--steps 25] [--rounds 1]"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import os
import sys
from pathlib import Path

from aethrion.data_governance import GovernanceError, validate_governance_record

SPACE_ID = "vizyon2027/AETHRION-LLM"
DEFAULT_MODELS = ["code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document"]


_TRAINER_OK = re.compile(r"(?im)^\s*-\s*Trainer ok:\s*(True|False)\b")
_PROGRESS = re.compile(r"Progress: (\S+) step (\d+|None) \| complete: (True|False)")


def parse_progress(out):
    """Extract the Space's machine-readable progress line without guessing."""
    m = _PROGRESS.search(str(out))
    if not m:
        return None
    return {
        "status": m.group(1),
        "step": None if m.group(2) == "None" else int(m.group(2)),
        "complete": m.group(3) == "True",
    }


def parse_session_result(out):
    """Require a durable, parseable progress report; detect exhausted GPU quota."""
    rendered = str(out)
    match = _TRAINER_OK.search(rendered)
    progress = parse_progress(rendered)
    quota_exceeded = bool(re.search(
        r"ZeroGPUQuotaExceeded|(?:GPU|ZeroGPU)\\s+quota.{0,80}(?:exceed|exhaust|limit)|"
        r"(?:exceed|exhaust).{0,80}(?:GPU|ZeroGPU)\\s+quota",
        rendered, re.IGNORECASE,
    ))
    valid_progress = bool(
        progress
        and isinstance(progress.get("step"), int)
        and progress.get("step") >= 0
        and progress.get("status") in {
            "training_interrupted_not_released",
            "training_complete_not_released",
        }
    )
    return {
        "trainer_ok": bool(match and match.group(1) == "True" and valid_progress and not quota_exceeded),
        "progress": progress,
        "complete": bool(progress and progress.get("complete") is True),
        "quota_exceeded": quota_exceeded,
    }


def run(client, models, steps, rounds):
    summary = []
    for _ in range(rounds):
        for model in models:
            out = client.predict(model, steps, api_name="/training_session")
            ok = parse_session_result(out)["trainer_ok"]
            print(f"[{model}] {'OK' if ok else 'NOT OK'}\n{str(out)[:900]}\n", flush=True)
            summary.append((model, ok))
    return summary


def _tail(out, n=500):
    text = " ".join(str(out).split())
    return text[-n:]


def run_until_complete(client, models, steps, max_rounds, min_steps=25, max_consecutive_failures=3):
    """Resumable sessions per model until training_complete_not_released or max_rounds.
    Steps are halved after a failed session (shorter sessions fit the GPU window).
    Completion never means release."""
    results = {}
    quota_blocked = False
    for model in models:
        cur, state = steps, {"ok": False, "complete": False, "step": None, "rounds": 0}
        if quota_blocked:
            state["quota_exceeded"] = True
            state["skipped_reason"] = "Earlier model exhausted the shared ZeroGPU quota."
            results[model] = state
            continue
        failures = 0
        for _ in range(max_rounds):
            state["rounds"] += 1
            try:
                out = client.predict(model, cur, api_name="/training_session")
            except Exception as exc:
                detail = str(exc)
                print(f"[{model}] call failed: {type(exc).__name__}: {detail[:200]}", flush=True)
                if parse_session_result(detail)["quota_exceeded"]:
                    state["quota_exceeded"] = True
                    quota_blocked = True
                    break
                failures += 1
                cur = max(min_steps, cur // 2)
                if failures >= max_consecutive_failures:
                    print(f"[{model}] stopping after {failures} consecutive failed sessions", flush=True)
                    break
                continue
            parsed = parse_session_result(out)
            ok, prog = parsed["trainer_ok"], parsed["progress"] or {}
            print(f"[{model}] round {state['rounds']} ok={ok} progress={prog}", flush=True)
            if parsed["quota_exceeded"]:
                state["quota_exceeded"] = True
                quota_blocked = True
                print(f"[{model}] stopping immediately: shared ZeroGPU quota is exhausted.", flush=True)
                break
            if not ok:
                failures += 1
                print(f"[{model}] session output (no credentials): {_tail(out, 900)}", flush=True)
                cur = max(min_steps, cur // 2)
                if failures >= max_consecutive_failures:
                    print(f"[{model}] stopping after {failures} consecutive failed sessions", flush=True)
                    break
                continue
            failures = 0
            state.update(ok=True, step=prog.get("step"), complete=parsed["complete"])
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


TRAINABLE_MODELS = ("code", "research", "safety", "vision", "image", "asr", "tts", "vocoder", "3d", "document")
AUDIO_MODELS = {"asr", "tts", "vocoder"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_manifest_path(base: Path, relative: str) -> Path:
    """Resolve a manifest path and reject absolute paths, traversal, and symlink escapes."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("manifest path is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("absolute manifest paths are not allowed")
    base_resolved = base.resolve()
    resolved = (base_resolved / candidate).resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError("manifest path escapes its permitted data directory") from None
    return resolved


def _manifest_artifact_path(root: Path, relative: str) -> Path:
    """Resolve manifest-relative artifact paths from the shared manifests directory."""
    return _safe_manifest_path(Path(root) / "data" / "specialists" / "manifests", relative)


def validate_training_approvals(models, root: Path | None = None) -> None:
    """Fail closed unless every selected family has a complete, hash-bound human review."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    invalid = sorted(set(models) - set(TRAINABLE_MODELS))
    if invalid:
        raise RuntimeError("Unsupported specialist model(s): " + ", ".join(invalid))

    failures = []
    for model in models:
        manifest_path = root / "data" / "specialists" / "manifests" / f"{model}.json"
        governance_path = (
            root / "data" / "authored" / "audio" / f"GOVERNANCE_{model}.json"
            if model in AUDIO_MODELS
            else root / "data" / "authored" / model / "GOVERNANCE.json"
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            governance = json.loads(governance_path.read_text(encoding="utf-8"))
            if manifest.get("model_id") != model:
                raise ValueError("manifest model_id mismatch")
            for key in ("training_allowed", "owner_approval", "privacy_reviewed"):
                if manifest.get(key) is not True:
                    raise ValueError(f"manifest {key} is not true")
            validate_governance_record(governance, model_id=model)
            validate_governance_record(manifest.get("governance"), model_id=model)
            if manifest.get("governance") != governance:
                raise ValueError("embedded governance does not match the reviewed governance file")
            if model in AUDIO_MODELS and manifest.get("speaker_consent_reviewed") is not True:
                raise ValueError("manifest speaker_consent_reviewed is not true")

            sources = manifest.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ValueError("manifest has no source inventory")
            for source in sources:
                if source.get("allow_training") is not True:
                    raise ValueError(f"source {source.get('source_id', '<unknown>')} is not approved for training")
                if source.get("privacy_reviewed") is not True:
                    raise ValueError(f"source {source.get('source_id', '<unknown>')} lacks privacy review")
                if model in AUDIO_MODELS and source.get("speaker_consent_verified") is not True:
                    raise ValueError(f"audio source {source.get('source_id', '<unknown>')} lacks speaker-consent evidence")
                relative = source.get("source_path")
                expected = source.get("sha256")
                if not isinstance(relative, str) or not isinstance(expected, str):
                    raise ValueError("source path/hash is missing")
                source_file = _manifest_artifact_path(root, relative)
                if not source_file.is_file() or _sha256(source_file) != expected:
                    raise ValueError(f"source snapshot missing or hash mismatch: {relative}")

            for key in ("dataset_path", "dataset_sha256"):
                if not manifest.get(key):
                    raise ValueError(f"manifest {key} is missing")
            dataset_file = _manifest_artifact_path(root, manifest["dataset_path"])
            if not dataset_file.is_file() or _sha256(dataset_file) != manifest["dataset_sha256"]:
                raise ValueError("prepared dataset missing or hash mismatch")
            if manifest.get("tokenizer_path"):
                tokenizer_file = _manifest_artifact_path(root, manifest["tokenizer_path"])
                if not tokenizer_file.is_file() or _sha256(tokenizer_file) != manifest.get("tokenizer_sha256"):
                    raise ValueError("tokenizer missing or hash mismatch")
        except (OSError, ValueError, TypeError, KeyError, GovernanceError) as exc:
            failures.append(f"{model}: {exc}")

    if failures:
        raise RuntimeError(
            "Specialist training blocked by fail-closed approval/data-integrity gate:\\n- "
            + "\\n- ".join(failures)
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--until-complete", action="store_true")
    ap.add_argument("--preflight-only", action="store_true", help="check approval and artifact gates without connecting to the Space")
    a = ap.parse_args()
    try:
        validate_training_approvals(a.models)
    except RuntimeError as exc:
        if a.preflight_only:
            print("PRE-FLIGHT BLOCKED\\n" + str(exc), file=sys.stderr)
            return 0
        print(str(exc), file=sys.stderr)
        return 3
    if a.preflight_only:
        print("PRE-FLIGHT PASSED: selected specialist approvals and artifact hashes are valid; no training was started.")
        return 0
    token, user, pw = os.environ.get("HF_TOKEN"), os.environ.get("AETHRION_SPACE_USERNAME"), os.environ.get("AETHRION_SPACE_PASSWORD")
    if not (token and user and pw):
        print("Missing HF_TOKEN / AETHRION_SPACE_USERNAME / AETHRION_SPACE_PASSWORD in GitHub Actions secrets.", file=sys.stderr)
        return 2
    client = make_client(SPACE_ID, token, (user, pw))
    if a.until_complete:
        res = run_until_complete(client, a.models, a.steps, a.rounds)
        print("SUMMARY", res)
        # A successful bounded session is not equivalent to completed training.
        # Fail the workflow if any model exhausted its rounds before completion.
        return 0 if all(v["ok"] and v["complete"] for v in res.values()) else 1
    results = run(client, a.models, a.steps, a.rounds)
    failed = [m for m, ok in results if not ok]
    print("SUMMARY", {"ok": [m for m, ok in results if ok], "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
