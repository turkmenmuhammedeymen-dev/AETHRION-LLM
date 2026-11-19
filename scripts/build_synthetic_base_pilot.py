"""Create an explicitly synthetic, unapproved Turkish BASE/CHAT pilot corpus.

The generated records are original, generic educational text. They are not
owner-authored evidence, licensed third-party data, or approval to train.
Outputs are written under data/raw (kept out of Git) and every source remains
allow_training=false until a responsible human completes the review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = {
    "fundamentals": [
        "Bir algoritma, bir problemi çözmek için izlenen açık ve sonlu adımlar bütünüdür.",
        "Bir programdaki değişken, çalışma sırasında değeri değişebilen isimlendirilmiş veri alanıdır.",
        "Fonksiyonlar, tekrar kullanılabilir işlemleri anlamlı bir isim altında toplar.",
        "Bir döngü, koşul sağlandığı sürece belirli komutların yeniden çalışmasını sağlar.",
        "Koşullu ifadeler, programın hangi yolu izleyeceğini bir mantık sonucuna göre belirler.",
        "Bir listenin sırası önemlidir; elemanlara konum numarası üzerinden erişilebilir.",
        "Sözlük yapısı, anahtarları değerlerle eşleştirerek veriye isim üzerinden ulaşmayı sağlar.",
        "Bir hata ayıklama süreci, beklenen sonuç ile gerçek davranış arasındaki farkı araştırır.",
        "Birim testleri, küçük yazılım parçalarının belirli girdilerde doğru sonuç verdiğini denetler.",
        "Sürüm kontrolü, dosyalardaki değişiklikleri geçmişleriyle birlikte izlemeyi mümkün kılar.",
        "Bir API, yazılım bileşenlerinin belirlenmiş kurallarla iletişim kurmasını sağlayan arayüzdür.",
        "JSON, yapılandırılmış veriyi anahtar ve değer çiftleriyle temsil eden metin biçimidir.",
        "UTF-8, farklı dillerdeki karakterleri kodlamak için kullanılan Unicode tabanlı kodlamadır.",
        "Bir veritabanı tablosu, benzer türde kayıtları satır ve sütunlar halinde düzenler.",
        "Birincil anahtar, tablodaki her kaydı diğerlerinden ayıran benzersiz alanı belirtir.",
        "Önbellek, sık kullanılan verilere daha hızlı erişmek için geçici kopyalar saklar.",
        "Asenkron işlem, bekleme süresinde başka işlerin ilerlemesine izin veren çalışma biçimidir.",
        "Bir istisna, programın normal akışını kesen ve ele alınması gereken çalışma zamanı durumudur.",
        "Kod biçimlendirme kuralları, ekip içinde okunabilirliği ve tutarlılığı artırır.",
        "Bir bağımlılık, uygulamanın çalışmak için kullandığı harici paket veya bileşendir.",
        "Şifre özeti, verinin belirli algoritmayla hesaplanan sabit uzunluklu parmak izidir.",
        "SHA-256 özeti, dosyanın aktarım veya saklama sonrasında değişip değişmediğini denetlemeye yardım eder.",
        "Bir günlük kaydı, uygulamanın olaylarını ve hata ayrıntılarını sonradan incelemek için saklar.",
        "Girdi doğrulama, uygulamanın beklenmeyen veya hatalı veriyi işlemesini önlemeye yardımcı olur.",
        "En az ayrıcalık ilkesi, her bileşene yalnızca gerekli izinlerin verilmesini önerir.",
        "Bir yazılım lisansı, kodun hangi koşullarda kullanılabileceğini ve paylaşılabileceğini açıklar.",
        "Açık kaynak kodu, lisans koşulları çerçevesinde incelenebilir ve değiştirilebilir.",
        "Bir derleyici, kaynak kodunu başka bir çalıştırılabilir biçime dönüştürür.",
        "Bir yorumlayıcı, program komutlarını çalışma sırasında işleyerek yürütür.",
        "Bellek kullanımı, uygulamanın çalışırken ayırdığı kaynakların izlenmesiyle değerlendirilebilir.",
        "Performans ölçümü, iyileştirme öncesi ve sonrası sonuçları aynı koşullarda karşılaştırmalıdır.",
        "Bir ağ isteği, istemcinin sunucudan veri veya işlem talep etmesini sağlar.",
        "HTTP durum kodları, sunucunun isteğe verdiği yanıtın genel sonucunu belirtir.",
        "Kimlik doğrulama, bir kullanıcının veya hizmetin kim olduğunu kontrol eder.",
        "Yetkilendirme, doğrulanmış kimliğin hangi işlemleri yapabileceğini belirler.",
        "Yedekleme, beklenmeyen veri kaybına karşı geri dönüş olanağı sağlar."
    ],
    "reasoning": [
        "Bir iddiayı değerlendirirken önce onu destekleyen kanıtları ve varsayımları birbirinden ayır.",
        "Bir deneyde tek değişkeni değiştirmek, sonucun hangi etkenden kaynaklandığını anlamayı kolaylaştırır.",
        "Ölçüm sonuçları belirsizlik içerebilir; bu nedenle yöntem ve hata payı açıklanmalıdır.",
        "Bir grafiğin eksenleri incelenmeden yalnızca çizginin görünümüne bakarak sonuç çıkarılmamalıdır.",
        "Bir kaynak güvenilir görünse bile yayın tarihi, yazar bilgisi ve dayanakları ayrıca kontrol edilmelidir.",
        "Korelasyon, iki değişkenin birlikte değiştiğini gösterir; tek başına nedensellik kanıtlamaz.",
        "Bir hipotez, gözlem ve deneylerle sınanabilecek açık bir önerme biçiminde kurulmalıdır.",
        "Bir problemin sınırlarını tanımlamak, gereksiz çözüm yollarını erken aşamada elemek için yararlıdır.",
        "Bir karar tablosu, farklı koşulların olası sonuçlarını düzenli biçimde karşılaştırır.",
        "Bir tahminin güven düzeyi, kullanılan veri miktarı ve yöntemin sınırlamalarıyla birlikte açıklanmalıdır.",
        "Bir hesaplamada birimlerin korunması, sayısal sonucun fiziksel anlamını denetlemeye yardımcı olur.",
        "Ortalama tek başına dağılımı anlatmaz; medyan ve yayılım ölçüleri de incelenebilir.",
        "Örneklem seçimi yanlıysa büyük veri miktarı bile hatalı genellemelere yol açabilir.",
        "Bir karşılaştırmada aynı ölçütlerin ve benzer koşulların kullanılması adil değerlendirme sağlar.",
        "Bir hata tekrar üretilebiliyorsa adımların kaydedilmesi kök nedeni bulmayı kolaylaştırır.",
        "Bir çözümün doğruluğu kadar maliyeti, güvenliği ve sürdürülebilirliği de değerlendirilmelidir.",
        "Bir kaynakta eksik bilgi varsa boşluğu tahminle doldurmak yerine belirsizlik açıkça belirtilmelidir.",
        "Bir araştırma sorusu, ölçülebilir ve kapsamı yönetilebilir olacak biçimde daraltılmalıdır.",
        "Bir modelin eğitim başarısı, görülmemiş örneklerdeki performansından ayrı değerlendirilmelidir.",
        "Eğitim ve test verisinin ayrılması, modelin ezber yerine genelleme yapıp yapmadığını anlamaya yardım eder.",
        "Veri sızıntısı, test bilgisinin eğitim sürecine karışmasıyla ölçüm sonuçlarını yanıltabilir.",
        "Bir sonuç beklenenden farklıysa önce veri kalitesi, yöntem ve ölçüm koşulları kontrol edilmelidir.",
        "Bir planın risk analizi, olası sorunları ve bunlara verilecek yanıtları önceden tanımlar.",
        "Bir öncelik listesi, sınırlı zamanı etkisi yüksek işlere ayırmayı kolaylaştırır.",
        "Bir süreçte darboğazı bulmak için her aşamanın süresi ayrı ayrı ölçülmelidir.",
        "Bir iddianın yanlışlanabilir olması, hangi gözlemin onu geçersiz kılacağını açıklayabilmeyi gerektirir.",
        "Farklı açıklamalar aynı gözlemi açıklıyorsa ayırt edici yeni bir deney tasarlanmalıdır.",
        "Bir teknik rapor, kullanılan yöntemi başka bir kişinin tekrar edebileceği açıklıkta anlatmalıdır.",
        "Bir veri kümesindeki eksik değerler sessizce sıfır kabul edilmemeli, etkileri incelenmelidir.",
        "Bir güvenlik değerlendirmesi, yalnızca olasılığı değil olası etkinin büyüklüğünü de dikkate alır.",
        "Bir öneri sunarken kanıt, varsayım ve kişisel değerlendirme ayrı biçimde ifade edilmelidir.",
        "Bir başarısız deney de yöntem ve gözlemler doğru kaydedildiğinde öğrenme sağlayabilir.",
        "Bir sistemin sınırlarını bilmek, ona uygun olmayan görevleri yüklemeyi önler.",
        "Bir kararın geri alınabilir olması, belirsizlik yüksekken küçük adımlarla ilerlemeyi destekler.",
        "Bir ölçüt iyileşirken başka bir ölçüt kötüleşiyorsa ödünleşim açıkça raporlanmalıdır.",
        "Bir dokümanın güncel olup olmadığı, sürüm numarası ve son değişiklik tarihiyle kontrol edilebilir."
    ],
    "digital_literacy": [
        "Güçlü bir parola, farklı hesaplarda tekrar kullanılmamalı ve güvenli biçimde saklanmalıdır.",
        "İki aşamalı doğrulama, parola dışında ek bir kontrol katmanı sağlayabilir.",
        "Şüpheli bir bağlantıya tıklamadan önce alan adı dikkatle incelenmelidir.",
        "Bir e-postadaki acil işlem baskısı, dolandırıcılık girişimlerinde kullanılan yaygın bir işarettir.",
        "Kişisel veriler yalnızca gerekli olduğunda ve uygun izinlerle paylaşılmalıdır.",
        "Bir uygulamaya verilen izinler, kullanım amacıyla düzenli olarak karşılaştırılmalıdır.",
        "Herkese açık kablosuz ağlarda hassas işlemler için ek güvenlik önlemleri düşünülmelidir.",
        "Yazılım güncellemeleri, bilinen güvenlik açıklarının kapatılmasına yardımcı olabilir.",
        "Yedeklerin düzenli aralıklarla geri yüklenerek denenmesi, çalıştıklarını doğrular.",
        "Bir dosyanın uzantısı tek başına içeriğinin güvenli olduğunu kanıtlamaz.",
        "Bilinmeyen göndericilerden gelen ekler açılmadan önce doğrulanmalıdır.",
        "Bir web sitesinde HTTPS bulunması bağlantıyı şifreler, fakat içeriğin doğru olduğunu garanti etmez.",
        "Hesap kurtarma seçenekleri güncel tutulmalı ve başkalarıyla paylaşılmamalıdır.",
        "Gizlilik ayarları, hangi bilgilerin kimlerle paylaşılabildiğini anlamaya yardımcı olur.",
        "Bir çevrim içi hizmetin veri politikasında saklama ve silme koşulları incelenmelidir.",
        "Açık oturumlar, ortak kullanılan cihazlardan ayrılmadan önce kapatılmalıdır.",
        "Bir dosyayı paylaşırken bağlantının erişim kapsamı ve süresi kontrol edilmelidir.",
        "Kaynağı bilinmeyen programlar, güvenilirliği doğrulanmadan çalıştırılmamalıdır.",
        "Sosyal medyada görülen bir haber, paylaşılmadan önce bağımsız kaynaklardan doğrulanmalıdır.",
        "Bir görselin bağlamı değiştirilebilir; tarih ve ilk yayın kaynağı araştırılmalıdır.",
        "Yapay zekâ çıktıları hata içerebilir ve önemli kararlar için bağımsız biçimde doğrulanmalıdır.",
        "Bir sohbet robotuna parola, kimlik numarası veya özel kayıtlar gönderilmemelidir.",
        "Dijital ayak izi, çevrim içi etkinliklerin zaman içinde bıraktığı izlerin bütünüdür.",
        "Bir hesabın ele geçirildiğinden şüphelenilirse güvenilir cihazdan parola yenilenmelidir.",
        "Kamuya açık depolarda erişim anahtarı veya gizli yapılandırma dosyası tutulmamalıdır.",
        "Bir erişim belirteci, yalnızca gerekli kapsam ve süreyle sınırlandırılmalıdır.",
        "İzin verilmeyen yazılım kopyalarını indirmek hukuki ve güvenlik sorunları doğurabilir.",
        "Bir çevrim içi tartışmada kişisel saldırı yerine iddia ve kanıt ele alınmalıdır.",
        "Dijital içerik üretirken kullanılan görsel, ses ve metinlerin lisansı kontrol edilmelidir.",
        "Bir dosyanın karma değeri, iki kopyanın aynı olup olmadığını karşılaştırmaya yarar.",
        "Güvenli geliştirme, gizli bilgileri kaynak kodundan ayrı tutmayı gerektirir.",
        "Bir hizmetin yedekleme politikası, veri kaybı durumunda geri dönüş süresini etkiler.",
        "Bir cihaz kaybolduğunda hesap oturumlarını uzaktan kapatma seçenekleri incelenmelidir.",
        "Bir web formu yalnızca gerekli alanları istemeli ve veriyi korumalıdır.",
        "Bir bildirimdeki bağlantı yerine hizmetin bilinen adresini kendin yazarak açmak daha güvenlidir.",
        "Bir güvenlik olayı yaşandığında tarih, etkilenen sistem ve alınan önlemler kaydedilmelidir."
    ],
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/synthetic_base_pilot")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for group_id, records in GROUPS.items():
        name = f"{group_id}.txt"
        target = root / name
        target.write_text("\n".join(records) + "\n", encoding="utf-8")
        sources.append({
            "source_id": f"synthetic-{group_id}",
            "path": name,
            "license": "AETHRION-SYNTHETIC-ORIGINAL-PENDING-HUMAN-REVIEW",
            "permitted_uses": "internal pilot preparation; training pending explicit human review",
            "allow_training": False,
            "origin": "ai_generated_synthetic",
            "generator": "AETHRION synthetic pilot authoring script",
            "record_count": len(records),
        })
    manifest = {
        "format": "aethrion-source-manifest-v1",
        "model": "base",
        "dataset_id": "aethrion-synthetic-base-pilot-v1",
        "dataset_type": "synthetic_only_not_owner_authored",
        "training_allowed": False,
        "sources": sources,
    }
    (root / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(len(rows) for rows in GROUPS.values())
    print(json.dumps({
        "output": str(root),
        "records": total,
        "groups": len(GROUPS),
        "training_allowed": False,
        "status": "prepared_for_human_review_only",
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
