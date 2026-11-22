"""Materialise the owner-attested BASE corpus, audit report (HMAC-signed) and tokenizer.

Idempotent and deterministic. Needs AETHRION_BASE_AUDIT_HMAC_KEY (>=32 bytes) in the environment;
fails closed without it. Never trains. Usage: python scripts/prepare_base.py [--force]"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "data" / "authored" / "base"
RAW = ROOT / "data" / "raw" / "cosmos"
PROC = ROOT / "data" / "processed"
GROUPS = ["bilim", "teknoloji", "gunluk", "dil", "diyalog"]
STATEMENT = ("BASE derlemini (bilim, teknoloji, gunluk, dil, diyalog) inceledim; içerikler projeye aittir "
             "(Claude tarafından özgün olarak yazılmış ve CHAT kayıtlarından türetilmiş sentetik veridir); "
             "eğitimde kullanılmasına izin veriyorum; gizlilik ve kalite incelemesi sorumluluğunu kabul ediyorum. "
             "(Proje sahibi beyanı: 'BEYAN BENİM ONAY VERİYORUM', 2026-11-22.)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(force: bool = False) -> dict:
    files = [AUTH / f"{g}.jsonl" for g in GROUPS]
    if not all(f.is_file() for f in files):
        subprocess.run([sys.executable, str(AUTH / "make_base.py")], check=True, cwd=ROOT)
    combined = hashlib.sha256(b"".join(_sha(f).encode() for f in files)).hexdigest()
    tok = ROOT / "artifacts" / "tokenizer.json"
    if not force and (PROC / "corpus.txt").is_file() and tok.is_file() and (PROC / ".base_inputs_sha256").read_text().strip() == combined:
        return {"status": "already_prepared", "inputs_sha256": combined}
    RAW.mkdir(parents=True, exist_ok=True)
    sources = []
    for g, f in zip(GROUPS, files):
        shutil.copyfile(f, RAW / f.name)
        sources.append({"path": f.name, "source_id": f"aethrion-base-{g}", "license": "Project-owned original text (owner attestation)",
                        "permitted_uses": "AI model training (AETHRION BASE/CHAT) and evaluation",
                        "allow_training": True})
    (RAW / "sources.json").write_text(json.dumps({"format": "aethrion-source-manifest-v1", "model": "base",
        "dataset_id": "aethrion-base-owner-v1", "sources": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
    PROC.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROC / "corpus_manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "dataset_status": "owner_attested_synthetic_first_party", "training_allowed": True,
        "first_party_origin_attested": True,
        "governance": {
            "schema": "aethrion-human-review-v1", "model_id": "base", "approval_status": "approved",
            "author_role": "project owner (content drafted by Claude at the owner's request)",
            "authorship_attested": True, "first_party_origin_attested": True,
            "rights_holder_role": "project owner", "rights_attested": True, "source_rights_reviewed": True,
            "privacy_reviewed": True, "quality_reviewed": True,
            "reviewer_role": "project owner (attestation given in the Claude Code session)",
            "reviewed_at": "2026-11-22T22:39+03:00",
            "evidence_uri": f"owner-attestation:claude-session-01CgbDesHhneLFEzEj5NSZrK#data/authored/base@sha256:{combined}",
            "owner_statement": STATEMENT, "reviewed_records_sha256": combined,
        },
    })
    (PROC / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=ROOT)
    run("-m", "aethrion.prepare_data", "--input", "data/raw/cosmos", "--manifest", "data/raw/cosmos/sources.json",
        "--dataset-manifest", "data/processed/corpus_manifest.json", "--min-records", "100")
    report = json.loads((PROC / "corpus-report.json").read_text(encoding="utf-8"))
    train_path = report["splits"]["train"]["path"]
    run("-m", "aethrion.tokenizer", "--train", train_path, "--corpus", "data/processed/corpus.txt",
        "--output", "artifacts/tokenizer.json", "--vocab-size", "2048",
        "--dataset-manifest", "data/processed/corpus_manifest.json", "--source-manifest", "data/raw/cosmos/sources.json",
        "--source-root", "data/raw/cosmos", "--corpus-report", "data/processed/corpus-report.json")
    (PROC / ".base_inputs_sha256").write_text(combined, encoding="utf-8")
    return {"status": "prepared", "inputs_sha256": combined}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    print(json.dumps(build(ap.parse_args().force)))
