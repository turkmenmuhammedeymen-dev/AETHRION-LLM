"""Append an authored round (module exposing R=[(prompt,response,topic),...]) to records.jsonl.
Skips duplicate prompts; ids/groups continue. Pure Python; does not touch any approval gate."""
import json, runpy, sys
from pathlib import Path
rec = Path("data/authored/chat/records.jsonl")
rows = [json.loads(l) for l in rec.read_text(encoding="utf-8").splitlines() if l.strip()]
seen = {" ".join(r["prompt"].lower().split()) for r in rows}
n = len(rows); added = skipped = 0; gseq = {}
for p, r, topic in runpy.run_path(sys.argv[1])["R"]:
    k = " ".join(p.lower().split())
    if k in seen: skipped += 1; continue
    seen.add(k); n += 1; added += 1
    gseq[topic] = gseq.get(topic, 0) + 1
    rows.append({"id": f"chat-{n:04d}", "group_id": f"chat-{topic}-{Path(sys.argv[1]).stem}-{(gseq[topic]-1)//3}", "prompt": p, "response": r})
rec.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(f"added={added} skipped_dupes={skipped} total={len(rows)}")
