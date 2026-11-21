"""Render authored records as a human-readable review sheet (read-only; sets no approval)."""
import json, sys
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
lines = [f"# Review sheet: {src}\n", f"{len(rows)} records. Mark problems by id; nothing here approves training.\n"]
for r in rows:
    body = "\n".join(f"- **{k}**: {v}" for k, v in r.items() if k not in ("id", "group_id"))
    lines.append(f"### {r['id']} ({r['group_id']})\n{body}\n")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
print(out, len(rows))
