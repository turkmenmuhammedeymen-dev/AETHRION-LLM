import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_base_pilot_is_labeled_and_unapproved(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "synthetic"
    subprocess.run(
        [sys.executable, str(root / "scripts" / "build_synthetic_base_pilot.py"),
         "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "sources.json").read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "synthetic_only_not_owner_authored"
    assert manifest["training_allowed"] is False
    assert len(manifest["sources"]) == 3
    total = 0
    for source in manifest["sources"]:
        assert source["origin"] == "ai_generated_synthetic"
        assert source["allow_training"] is False
        rows = (output / source["path"]).read_text(encoding="utf-8").splitlines()
        assert len(rows) == source["record_count"]
        assert all(len(row) >= 20 for row in rows)
        total += len(rows)
    assert total >= 100
