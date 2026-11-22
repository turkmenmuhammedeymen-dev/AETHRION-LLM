"""Record an owner attestation for authored datasets, bound to the exact records hash.
Run ONLY after the owner has given the statement. Usage:
  python3 scripts/apply_owner_attestation.py --session SESSION --statement "..." --safety-reviewed model [model ...]
Writes data/authored/<m>/GOVERNANCE.json and updates data/specialists/manifests/<m>.json approval flags."""
import argparse, hashlib, json, datetime
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+"); ap.add_argument("--session", required=True)
ap.add_argument("--statement", required=True); ap.add_argument("--safety-reviewed", action="store_true")
a = ap.parse_args()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).isoformat(timespec="minutes")
for m in a.models:
    src = Path(f"data/authored/{m}/records.jsonl"); h = hashlib.sha256(src.read_bytes()).hexdigest()
    mf = Path(f"data/specialists/manifests/{m}.json"); man = json.loads(mf.read_text(encoding="utf-8"))
    if hashlib.sha256(Path(f"data/specialists/manifests/{m}/source.jsonl").read_bytes()).hexdigest() != h:
        raise SystemExit(f"{m}: prepared source.jsonl does not match records.jsonl — re-run specialist_prepare")
    gov = {"schema": "aethrion-human-review-v1", "model_id": m, "approval_status": "approved",
           "author_role": "project owner (content drafted by Claude at the owner's request)",
           "authorship_attested": True, "first_party_origin_attested": True, "rights_holder_role": "project owner",
           "rights_attested": True, "source_rights_reviewed": True, "privacy_reviewed": True, "quality_reviewed": True,
           "reviewer_role": "project owner (attestation given in the Claude Code session)", "reviewed_at": now,
           "evidence_uri": f"owner-attestation:claude-session-{a.session}#data/authored/{m}/records.jsonl@sha256:{h}",
           "speaker_consent_reviewed": False, "safety_reviewed": bool(a.safety_reviewed and m == "safety"),
           "owner_statement": a.statement, "reviewed_records_sha256": h}
    Path(f"data/authored/{m}/GOVERNANCE.json").write_text(json.dumps(gov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    man["governance"] = {k: v for k, v in gov.items() if k not in ("owner_statement", "reviewed_records_sha256")}
    man.update(training_allowed=True, owner_approval=True, privacy_reviewed=True,
               dataset_status="owner_attested_pilot_not_released")
    for s in man["sources"]:
        s["allow_training"] = True; s["privacy_reviewed"] = True; s["evidence_uri"] = gov["evidence_uri"] if False else s["evidence_uri"]
    mf.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(m, "attested", h[:12])
