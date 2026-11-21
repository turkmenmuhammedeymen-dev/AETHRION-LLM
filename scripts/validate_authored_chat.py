"""Validate owner-authored CHAT records (pure Python). Never edits gates or approves data."""
import json, re, sys, hashlib
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/authored/chat/records.jsonl")
FIELDS = (sys.argv[2] if len(sys.argv) > 2 else "id,group_id,prompt,response").split(",")
PII = [re.compile(p, re.I) for p in (r"[\w.+-]+@[\w-]+\.[\w.]+", r"\+?\d[\d\s().-]{9,}\d",
       r"\b(?:sk|hf|ghp|AKIA)[-_A-Za-z0-9]{12,}", r"https?://")]
errors, seen_ids, seen_prompts, groups = [], set(), set(), {}
raw = PATH.read_bytes()
for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        errors.append(f"{n}: invalid JSON"); continue
    for k in FIELDS:
        if not isinstance(r.get(k), str) or not r[k].strip():
            errors.append(f"{n}: missing/empty {k}")
    if r.get("id") in seen_ids: errors.append(f"{n}: duplicate id")
    seen_ids.add(r.get("id"))
    key = " ".join(str(r.get("prompt", r.get("text", r.get("query", "")))).lower().split())
    if key in seen_prompts: errors.append(f"{n}: duplicate prompt")
    seen_prompts.add(key)
    for rx in PII:
        if rx.search(" ".join(str(r.get(f, "")) for f in FIELDS if f not in ("id", "group_id"))):
            errors.append(f"{n}: possible personal data/secret/URL ({rx.pattern[:20]})")
    groups.setdefault(r.get("group_id"), 0); groups[r.get("group_id")] += 1
print(json.dumps({"records": len(seen_ids), "groups": len(groups), "sha256": hashlib.sha256(raw).hexdigest(),
                  "errors": errors}, ensure_ascii=False, indent=2))
sys.exit(1 if errors else 0)
