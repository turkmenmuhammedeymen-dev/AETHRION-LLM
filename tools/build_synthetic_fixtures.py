#!/usr/bin/env python3
"""Generate explicitly synthetic, non-training fixtures for all 11 AETHRION models."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SCHEMA = "aethrion.synthetic-fixtures.v1"
SAMPLES = {
 "chat": [("conversation","Bir algoritma nedir?","Bir problemi çözmek için izlenen sonlu ve sıralı adımlardır.","text"),("conversation","Yedek bütünlüğü nasıl doğrulanır?","Bir sağlama toplamı hesaplanır ve geri yükleme testi yapılır.","text")],
 "code": [("code_generation","Python'da listenin uzunluğunu döndüren fonksiyon yaz.","def item_count(items):\n    return len(items)","text"),("code_explanation","json.dumps ne yapar?","Python nesnesini JSON metnine serileştirir.","text")],
 "vision": [("image_understanding",{"asset_id":"placeholder-image-001","payload":None},"Görüntü verisi yok; içerik belirlenemez.","image"),("visual_question_answering",{"asset_id":"placeholder-image-002","payload":None,"question":"Ne görüyorsun?"},"Görüntü verisi yok; yanıt üretilemez.","image")],
 "image": [("image_spec","Mavi geometrik şekillerden sade poster tarifi.",{"prompt":"Sade, mavi geometrik şekiller içeren eğitim posteri.","rendered_asset":None},"image"),("image_edit_spec",{"asset_id":"placeholder-image-003","payload":None,"instruction":"Arka planı sadeleştir."},{"edit_plan":"Arka planı sadeleştir; ana özneyi koru.","rendered_asset":None},"image")],
 "asr": [("speech_recognition",{"asset_id":"placeholder-audio-001","payload":None},None,"audio"),("speech_recognition",{"asset_id":"placeholder-audio-002","payload":None},None,"audio")],
 "tts": [("text_to_speech","Merhaba. AETHRION sistem testi.",{"audio_asset":None,"speaker_consent":None},"audio"),("text_to_speech","İşlem tamamlandı.",{"audio_asset":None,"speaker_consent":None},"audio")],
 "vocoder": [("vocoder",{"mel_asset":None},{"audio_asset":None},"audio"),("vocoder",{"mel_asset":None},{"audio_asset":None},"audio")],
 "3d": [("text_to_3d","Basit, alçak çokgenli masa lambası.",{"geometry_asset":None,"format":"glb"},"3d"),("pointcloud_understanding",{"pointcloud_asset":None},None,"3d")],
 "research": [("evidence_summary","Yöntem ve sonuç bölümleri neden ayrılır?","Yöntem nasıl çalışıldığını, sonuçlar gözlenen bulguları açıklar.","text"),("uncertainty","Kaynağı olmayan iddia nasıl sunulmalı?","Doğrulanmamış olduğu belirtilmeli; kesin bilgi gibi aktarılmamalıdır.","text")],
 "document": [("document_outline","Proje raporu için bölüm planı.",["Amaç","Kapsam","Yöntem","Bulgular","Sınırlılıklar","Sonuç"],"text"),("field_extraction","Toplantı tarihi: 18 Kasım 2026. Konu: veri doğrulama.",{"date":"18 Kasım 2026","topic":"veri doğrulama"},"text")],
 "safety": [("safe_completion","Web uygulamasında hesap güvenliği nasıl artırılır?","MFA, güçlü benzersiz parolalar, güncel yazılım ve güvenli oturum yönetimi kullanın.","text"),("safe_completion","Kaynağı doğrulamadan iddia paylaşmalı mıyım?","Hayır. Güvenilir kaynaklarla doğrulayın ve belirsizliği belirtin.","text")],
}
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",default="data/synthetic/dev_fixtures"); a=p.parse_args()
 root=Path(a.output); root.mkdir(parents=True,exist_ok=True); models={}; total=0
 for model,samples in SAMPLES.items():
  rows=[]
  for i,(task,inp,out,modality) in enumerate(samples,1):
   row={"schema_version":SCHEMA,"id":f"synthetic-{model}-{i:04d}","model_id":model,"task":task,"input":inp,"expected_output":out,"modality":modality,"source_kind":"ai_generated_synthetic","first_party_origin_attested":False,"human_reviewed":False,"training_eligible":False}
   if modality in {"image","audio","3d"}: row["payload_status"]="missing"
   if model=="safety": row["label_status"]="synthetic_unreviewed"
   row["record_sha256"]=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   rows.append(row)
  target=root/f"{model}.jsonl"; target.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
  models[model]={"records":len(rows),"file":target.name,"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"training_eligible":False}
  total+=len(rows)
 manifest={"schema_version":SCHEMA,"project":"AETHRION-LLM","dataset_status":"synthetic_development_fixtures_only","purpose":"schema_parser_and_fail_closed_pipeline_tests","training_allowed":False,"first_party_origin_attested":False,"human_reviewed":False,"production_data":False,"total_records":total,"models":models,"limitations":["Synthetic examples are not owner-authored real-world source data.","Image, audio and 3D records have metadata placeholders only; no media payloads exist.","SAFE labels are synthetic and have not received independent human review.","This manifest must never unlock production training."]}
 (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(f"Created {total} synthetic development fixtures for {len(models)} models.")
 print("training_allowed=False")
if __name__=="__main__": main()
