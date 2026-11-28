import json
from scripts.apply_release_decisions import apply


def _template(path, names):
    path.write_text(json.dumps({"benchmarks": {n: {"direction": d, "value": None, "passed": False} for n, d in names}}))


def test_unmeasured_never_approved_and_threshold_enforced(tmp_path):
    _template(tmp_path / "safety_evaluation.json", [("calibration_ece", "min"), ("false_negative_rate", "min")])
    _template(tmp_path / "code_evaluation.json", [("unit_test_pass_rate", "max")])
    audit = {"models": [{"model": "safety", "evaluation": {"task_metrics": {"calibration_ece": 0.08, "false_negative_rate": 0.107}}}]}
    dec = {"owner": "Owner", "models": {
        "safety": {"karar": "yayin", "esikler": {"calibration_ece": 0.1, "false_negative_rate": 0.05}},
        "code": {"karar": "yayin", "esikler": {"unit_test_pass_rate": 0.6}}}}
    out = apply(dec, audit, tmp_path, now="t")
    assert out["safety"].startswith("NOT approved") and "0.107" in out["safety"]
    assert out["code"].startswith("NOT approved") and "no measurement" in out["code"]
    assert json.loads((tmp_path / "safety_evaluation.json").read_text()).get("status") != "approved_for_release"


def test_approves_only_when_all_measured_and_met(tmp_path):
    _template(tmp_path / "safety_evaluation.json", [("calibration_ece", "min"), ("false_negative_rate", "min")])
    audit = {"models": [{"model": "safety", "evaluation": {"task_metrics": {"calibration_ece": 0.08, "false_negative_rate": 0.107}}}]}
    dec = {"owner": "Owner", "models": {"safety": {"karar": "yayin", "esikler": {"calibration_ece": 0.1, "false_negative_rate": 0.2}}}}
    assert apply(dec, audit, tmp_path, now="t")["safety"].startswith("approved_for_release")
    rec = json.loads((tmp_path / "safety_evaluation.json").read_text())
    assert rec["status"] == "approved_for_release" and rec["reviewer"] == "Owner" and len(rec["approval_evidence_sha256"]) == 64
