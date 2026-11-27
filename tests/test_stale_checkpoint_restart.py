import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from aethrion import space_session as ss
from aethrion.train_family import _run_forwarding_stop_signals
from tests.test_artifact_sync import ENV, Fake

STALE = "ValueError: Specialist source/approval manifest changed; refusing unsafe resume."


def _setup(root: Path) -> Path:
    cfg = root / "configs" / "specialists"
    cfg.mkdir(parents=True)
    (cfg / "code.json").write_text(json.dumps({"model_id": "code", "output_dir": "artifacts/models/code"}))
    out = root / "artifacts" / "models" / "code"
    out.mkdir(parents=True)
    (out / "checkpoint_last.pt").write_bytes(b"stale")
    return out


def _fresh_trainer(out: Path, calls: list):
    def run(cmd, **_):
        calls.append(cmd)
        if (out / "checkpoint_last.pt").is_file():
            return SimpleNamespace(returncode=1, stdout="", stderr=STALE)
        (out / "checkpoint_last.pt").write_bytes(b"fresh")
        (out / "training_report.json").write_text(json.dumps(
            {"status": "training_interrupted_not_released", "steps": 3}))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    return run


def test_stale_checkpoint_is_reported_not_trained_by_default(tmp_path):
    out = _setup(tmp_path)
    calls = []
    r = ss.run_session("code", 3, root=tmp_path, api=Fake(), environ=ENV, runner=_fresh_trainer(out, calls))
    assert not r["trainer_ok"] and r["pushed"] == [] and not r["released"]
    assert r["progress"]["status"] == "stale_checkpoint_incompatible_not_released"
    assert len(calls) == 1 and (out / "checkpoint_last.pt").read_bytes() == b"stale"


def test_fresh_on_stale_quarantines_and_restarts_once(tmp_path):
    out = _setup(tmp_path)
    calls = []
    r = ss.run_session("code", 3, root=tmp_path, api=Fake(), environ=ENV,
                       runner=_fresh_trainer(out, calls), fresh_on_stale=True)
    assert r["trainer_ok"] and r["stale_checkpoint_replaced"] == ["checkpoint_last.pt"]
    assert r["progress"]["step"] == 3 and not r["released"]
    assert len(calls) == 2 and (out / "checkpoint_last.pt").read_bytes() == b"fresh"
    kept = list((tmp_path / "artifacts" / "models" / ".stale").glob("code-*/checkpoint_last.pt"))
    assert len(kept) == 1 and kept[0].read_bytes() == b"stale"


def test_non_stale_failure_is_not_retried(tmp_path):
    out = _setup(tmp_path)
    calls = []

    def run(cmd, **_):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="RuntimeError: out of memory")
    r = ss.run_session("code", 3, root=tmp_path, api=Fake(), environ=ENV, runner=run, fresh_on_stale=True)
    assert not r["trainer_ok"] and len(calls) == 1 and (out / "checkpoint_last.pt").is_file()


def test_family_runner_forwards_sigterm_to_trainer(tmp_path):
    marker = tmp_path / "got_sigterm"
    script = (
        "import signal,sys,time,pathlib\n"
        f"p=pathlib.Path({str(marker)!r})\n"
        "signal.signal(signal.SIGTERM, lambda *a: (p.write_text('1'), sys.exit(0)))\n"
        "time.sleep(30)\n"
    )
    import threading, os
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    started = time.time()
    result = _run_forwarding_stop_signals([sys.executable, "-c", script], tmp_path)
    assert result.returncode == 0 and marker.read_text() == "1" and time.time() - started < 20
