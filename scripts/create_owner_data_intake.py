"""Create an empty, fail-closed intake workspace for all AETHRION models."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from aethrion.data_governance import review_template

SPECS = {
 "chat":("text","base",["id","group_id","prompt","response"],[],"First-party authored prompt/response pairs; reuse exact BASE tokenizer; no private chats."),
 "code":("code","code",["id","group_id","text"],[],"Original code and dependency/license provenance."),
 "vision":("vision","vision",["id","group_id","image_path","caption"],["images"],"Image origin, training rights, privacy and subject-consent review."),
 "image":("image","image",["id","group_id","image_path","caption"],["images"],"Rights-cleared image-caption pairs; privacy and memorization review."),
 "asr":("audio","asr",["id","speaker_id","wav_path","transcript"],["audio"],"Explicit speaker consent covering model training is mandatory."),
 "tts":("audio","tts",["id","speaker_id","wav_path","transcript","durations"],["audio"],"Speaker consent and documented token/frame alignment are mandatory."),
 "vocoder":("audio","vocoder",["id","speaker_id","wav_path"],["audio"],"Speaker consent for model training and speaker-group isolation are mandatory."),
 "3d":("3d","3d",["id","group_id","point_path","text"],["point_clouds"],"Point-cloud ownership, model-training rights and scan/CAD provenance."),
 "research":("text","research",["id","group_id","query","passage"],[],"First-party passages or explicit model-training permission; retain citations."),
 "document":("document","document",["id","group_id","text","boxes","layout_ids","tag_labels"],["documents"],"Rights-cleared documents; exclude personal and confidential data."),
 "safety":("text","safety",["id","group_id","text","decision","reasons"],[],"Human-assigned labels and separate safety review; benign prevention examples only."),
}
REQUIREMENTS=[
 "UTF-8 owner-authored records with documented authorship and rights-holder role",
 "explicit first-party origin attestation for the complete dataset",
 "timestamped human review of source rights, training permission, privacy and quality",
 "unique sample IDs, deduplication, and group-disjoint train/validation/test splits",
 "local source files with SHA-256 provenance",
]

def write_json(path, value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def create_intake_workspace(output="data/intake"):
 target=Path(output).expanduser()
 if target.exists() or target.is_symlink():
  raise FileExistsError(f"Refusing to overwrite existing intake path: {target}")
 target.parent.mkdir(parents=True,exist_ok=True)
 target.mkdir()
 rows=[]
 try:
  for model,(modality,governance_model,fields,assets,review) in SPECS.items():
   folder=target/model
   folder.mkdir()
   for asset in assets: (folder/asset).mkdir()
   (folder/"records.jsonl").write_text("",encoding="utf-8")
   write_json(folder/"governance.json",review_template(governance_model))
   submission={
    "format":"aethrion-data-intake-v1","model_id":model,"modality":modality,
    "status":"awaiting_owner_authored_records_and_human_review",
    "record_file":"records.jsonl","record_fields":fields,"asset_directories":assets,
    "source_count":0,"training_allowed":False,"owner_approval":False,
    "privacy_reviewed":False,"rights_reviewed":False,
    "speaker_consent_reviewed":False,"safety_reviewed":False,
    "governance_file":"governance.json","requirements":list(REQUIREMENTS),
    "model_specific_review":review,
   }
   if model in {"asr","tts","vocoder"}:
    submission["requirements"] += ["documented speaker consent covering model training"]
   if model=="safety":
    submission["requirements"] += ["separate human safety-data review; human-reviewed ALLOW/REVIEW/BLOCK labels"]
   write_json(folder/"submission.json",submission)
   rows.append({"model_id":model,"modality":modality,"record_count":0,"training_allowed":False})
  overview={"format":"aethrion-data-intake-workspace-v1","project":"AETHRION-LLM",
   "model_count":len(rows),"training_started":False,"automatic_approval":False,"models":rows,
   "notice":"Empty intake workspace only. No records, rights, consent, or training approval are included."}
  write_json(target/"workspace.json",overview)
  lines=["# AETHRION-LLM owner-authored data intake","",
   "This workspace is empty and is not a dataset. It cannot be used for training.",
   "Do not add copied, scraped, synthetic-fixture, private-chat, or third-party material without documented permission for this exact model-training use.",
   "Obtain responsible adult support for rights, privacy and consent review.",
   "Keep source files local; record origin, license, permitted uses and SHA-256.",
   "Use stable group IDs so the same speaker, repository, document or source group cannot cross data splits.",
   "Never guess or pre-check an approval. Preparation tools do not grant approval.","",
   "## Model folders and required JSONL fields",""]
  for model,(modality,_,fields,_,review) in SPECS.items():
   lines += [f"- **{model}** ({modality}): {', '.join(fields)}.",f"  - {review}"]
  lines += ["","ASR/TTS/Vocoder require explicit speaker consent for model training.",
   "SAFE labels require human review and benign, prevention-oriented examples.",
   "Do not add private messages, school/medical records, account details, or identifiable personal data.",
   "All submission flags are false and governance status is pending. This generator does not edit production manifests, download data, create tokenizers/checkpoints, or start training.",""]
  (target/"README.md").write_text("\n".join(lines),encoding="utf-8")
 except Exception:
  shutil.rmtree(target,ignore_errors=True)
  raise
 return overview

def main():
 parser=argparse.ArgumentParser(description="Create a blank, fail-closed 11-model intake workspace.")
 parser.add_argument("--output",default="data/intake",help="New output path; existing paths are never overwritten.")
 args=parser.parse_args()
 print(json.dumps(create_intake_workspace(args.output),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
