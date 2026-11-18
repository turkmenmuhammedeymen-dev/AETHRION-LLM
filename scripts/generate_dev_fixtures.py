"""Generate tiny, non-production AETHRION development fixtures.

These synthetic files exercise format readers only. They are not training data.
No approval flag is ever set to true.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

MODEL_IDS = ("chat","code","vision","image","asr","tts","3d","research","document","safety","vocoder")
TEXT = {
"chat":[("Bir dosyayı güvenli biçimde nasıl yedeklerim?","Önce kopya oluştur, SHA-256 özeti kaydet ve yedeği ayrı konumda doğrula."),("Python'da JSON nasıl okunur?","Dosyayı UTF-8 açıp json.load kullan; JSONDecodeError hatasını yakala."),("Bir iddiayı nasıl doğrularım?","Birincil kaynağı incele, tarihi kontrol et ve bağımsız kaynaklarla çapraz doğrula.")],
"code":["def normalize_name(value):\n    return ' '.join(value.strip().split()).casefold()","def safe_divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('b sıfır olamaz')\n    return a / b","def is_even(value):\n    return value % 2 == 0"],
"research":[("Bir yazılım kütüphanesinin bakım durumu nasıl değerlendirilir?","Son sürüm tarihi, güvenlik bildirimleri, kritik hatalar ve sürdürücü etkinliği incelenir."),("Bilimsel iddianın kanıt gücü nasıl incelenir?","Yöntem, örneklem, belirsizlik, çıkar çatışması ve bağımsız tekrarlar incelenmelidir."),("Veri kümesinin kullanım koşulları nasıl doğrulanır?","Sürüm, hak sahibi, lisans, atıf, ticari kullanım ve ML eğitimi hükümleri kontrol edilir.")],
"document":["Başlık: Sistem Durumu\nBölüm: Veri bütünlüğü\nDurum: İnceleme bekliyor","Başlık: Deney Raporu\nAmaç: Tekrarlanabilirliği ölçmek\nYöntem: Sabit tohum ve sürüm kaydı","Başlık: Proje Kararı\nKarar: Dağıtım ertelendi\nGerekçe: Değerlendirme kanıtı eksik"],
"safety":["Bir hesabın güvenliğini artırmak için hangi savunma adımlarını uygulamalıyım?","Bu isteğin amacı belirsiz; güvenli ve yasal kullanım bağlamını açıklar mısın?","Başkasının hesabına izinsiz erişmeye yardımcı olamam; hesap kurtarma adımlarını anlatabilirim."]
}
def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def ppm(path,index,size=64):
    path.parent.mkdir(parents=True,exist_ok=True); pix=bytearray()
    for y in range(size):
        for x in range(size):
            ring=int(math.hypot(x-size/2,y-size/2)//5)%2
            pix.extend(((x*3+index*31)%256,(y*4+index*17)%256,180 if ring else 70))
    path.write_bytes(f"P6\n{size} {size}\n255\n".encode("ascii")+pix)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="data/dev_fixtures"); root=Path(p.parse_args().output).resolve(); root.mkdir(parents=True,exist_ok=True)
    (root/"base.txt").write_text("\n".join(["AETHRION fikstürü: dosya bütünlüğü SHA-256 ile denetlenir.","AETHRION fikstürü: veri bölümleri kaynak bazında ayrılır.","AETHRION fikstürü: güvenlik etiketi insan incelemesi olmadan onaylanmaz."])+"\n",encoding="utf-8")
    for model, samples in TEXT.items():
        rows=[]
        for i,s in enumerate(samples,1):
            row={"id":f"{model}-{i:03d}","group_id":f"synthetic-{model}-{i}"}
            if model=="chat": row.update(prompt=s[0],response=s[1])
            elif model=="research": row.update(query=s[0],passage=s[1])
            else: row["text"]=s
            if model=="safety": row.update(label_status="unreviewed_proposal",decision=None,reasons=None)
            rows.append(row)
        write_jsonl(root/model/"records.jsonl",rows)
    for model in ("vision","image"):
        rows=[]
        for i in range(3):
            name=f"abstract-{i+1}.ppm"; ppm(root/model/"assets"/name,i)
            rows.append({"id":f"{model}-{i+1}","group_id":f"synthetic-{model}-{i+1}","image_path":f"assets/{name}","caption":"Sentetik geometrik renk deseni; gerçek dünya nesnesi değildir."})
        write_jsonl(root/model/"records.jsonl",rows)
    rows=[]
    for i in range(3):
        rng=np.random.default_rng(1000+i); pts=rng.uniform(-.8,.8,size=(64,3)).astype(np.float32)
        path=root/"3d"/"assets"/f"cloud-{i+1}.npy"; path.parent.mkdir(parents=True,exist_ok=True); np.save(path,pts,allow_pickle=False)
        rows.append({"id":f"3d-{i+1}","group_id":f"synthetic-3d-{i+1}","point_path":f"assets/{path.name}","text":"Sentetik rastgele nokta bulutu; anlamlı nesne geometrisi değildir."})
    write_jsonl(root/"3d"/"records.jsonl",rows)
    manifest={"format":"aethrion-dev-fixtures-v1","dataset_status":"synthetic_smoke_fixtures_only","training_allowed":False,"owner_approval":False,"privacy_reviewed":False,"human_reviewed":False,"production_data":False,"models":{}}
    for model in MODEL_IDS:
        ready=model in {"chat","code","vision","image","3d","research","document","safety"}
        manifest["models"][model]={"fixture_created":ready,"production_training_ready":False,"note":"No speech recordings are fabricated; real reviewed speech and explicit speaker permission are required." if model in {"asr","tts","vocoder"} else "Synthetic smoke fixture only; not a training corpus."}
    (root/"fixture_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (root/"README.md").write_text("# AETHRION development fixtures\n\nGenerated by scripts/generate_dev_fixtures.py.\n\nNOT TRAINING DATA. Tiny synthetic fixtures exercise format readers only. They are not owner-authored production data, are not human-reviewed, and must not be used for training, benchmark claims, or release evidence. All approvals remain false. SAFE labels are intentionally absent. Audio is not fabricated as speech. Image rasters are abstract patterns; 3D clouds are random points, not real objects.\n",encoding="utf-8")
    print(json.dumps({"output":str(root),"fixture_models":8,"audio_models_blocked":["asr","tts","vocoder"],"training_allowed":False},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
