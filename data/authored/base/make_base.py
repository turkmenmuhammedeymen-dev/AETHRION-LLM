"""Writes the first-party BASE corpus sources (owner-approved, Claude-drafted original Turkish text).

Four topic groups of original factual sentences (rolling 3-sentence paragraphs) plus a dialogue
group built from the already attested CHAT records. No external text is copied.
Run: python data/authored/base/make_base.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = {
"bilim": """Su, deniz seviyesinde yaklaşık yüz derecede kaynar ve sıfır derecede donar.
Güneş, Dünya'ya ışık ve ısı sağlayan yıldızdır.
Dünya, Güneş'in çevresindeki bir turunu yaklaşık üç yüz altmış beş gün altı saatte tamamlar.
Ay, Dünya'nın tek doğal uydusudur ve gelgitleri etkiler.
Bitkiler, fotosentez sırasında ışık enerjisini kullanarak karbondioksit ve sudan besin üretir.
Oksijen, canlıların solunumu için gerekli olan gazdır.
Hava, büyük ölçüde azot ve oksijen gazlarından oluşur.
Ses, titreşimlerin hava gibi bir ortamda yayılmasıyla oluşan dalgadır.
Işık boşlukta saniyede yaklaşık üç yüz bin kilometre hızla yayılır.
Yerçekimi, kütleye sahip cisimlerin birbirini çekmesine neden olan kuvvettir.
Bir elementin atom numarası, çekirdeğindeki proton sayısına eşittir.
Hidrojen, evrendeki en hafif ve en bol bulunan elementtir.
Bir hücre, canlıların yapı ve işlev bakımından en küçük birimidir.
DNA, canlıların kalıtsal bilgisini taşıyan çift sarmal yapıdaki moleküldür.
Kalp, kanı vücutta dolaştıran kasılan bir organdır.
Akciğerler, solunum sırasında kana oksijen verir ve karbondioksiti dışarı atar.
Mevsimler, Dünya'nın eksen eğikliği nedeniyle oluşur.
Gece ve gündüz, Dünya'nın kendi ekseni etrafında dönmesiyle oluşur.
Bir yıldız, merkezindeki füzyon tepkimeleriyle enerji üretir.
Samanyolu, Güneş Sistemi'ni de içeren bir gökadadır.
Merkür, Güneş'e en yakın gezegendir.
Jüpiter, Güneş Sistemi'ndeki en büyük gezegendir.
Satürn'ün çevresinde buz ve kaya parçalarından oluşan belirgin halkalar bulunur.
Kütle, bir cisimdeki madde miktarının ölçüsüdür.
Hız, bir cismin birim zamanda aldığı yoldur.
Isı, sıcaklık farkı nedeniyle bir cisimden diğerine geçen enerjidir.
Elektrik akımı, yük taşıyıcıların bir iletken içinde düzenli hareketidir.
Mıknatısın iki kutbu vardır ve aynı kutuplar birbirini iter.
Buharlaşma, bir sıvının gaz haline geçmesidir.
Yağmur, bulutlardaki su damlacıklarının birleşip ağırlaşarak yere düşmesidir.
Deprem, yer kabuğundaki enerjinin ani boşalmasıyla oluşan sarsıntıdır.
Volkanlar, yer altındaki erimiş kayaçların yüzeye çıktığı noktalardır.
Okyanuslar, Dünya yüzeyinin yaklaşık yüzde yetmişini kaplar.
Mercan resifleri, denizlerde pek çok canlıya yuva olan zengin ekosistemlerdir.
Ormanlar, karbondioksiti soğurarak iklimin dengelenmesine yardımcı olur.
Arılar, çiçekleri tozlaştırarak birçok bitkinin üremesine katkı sağlar.
Besin zinciri, canlılar arasındaki enerji aktarımını gösterir.
Bir ekosistem, canlıların ve çevrelerinin birlikte oluşturduğu bütündür.
Fosiller, geçmişte yaşamış canlıların kayaçlarda korunmuş izleridir.
Bilimsel yöntem, gözlem yapmayı, hipotez kurmayı ve deneyle sınamayı içerir.
Bir deneyde yalnızca bir değişkeni değiştirmek, sonucun nedenini anlamayı kolaylaştırır.
Mikroskop, gözle görülemeyen küçük yapıları büyüterek incelememizi sağlar.
Teleskop, uzaktaki gök cisimlerini daha yakın ve parlak göstermek için kullanılır.
Termometre sıcaklığı, barometre hava basıncını ölçer.
Karışım, birden fazla maddenin kimyasal bağ kurmadan bir araya gelmesidir.
Tuz suda çözündüğünde görünmez hale gelir ama suyun tadı değişir.
Metaller genellikle ısıyı ve elektriği iyi iletir.
Cam, kumun yüksek sıcaklıkta eritilip soğutulmasıyla elde edilir.
Geri dönüşüm, kullanılmış malzemelerin yeniden hammadde olarak değerlendirilmesidir.
Yenilenebilir enerji kaynakları arasında güneş, rüzgâr ve akarsu bulunur.""",
"teknoloji": """Bilgisayar, verileri alıp işleyen ve sonuç üreten bir makinedir.
Bir program, bilgisayara ne yapacağını anlatan komutlar dizisidir.
Algoritma, bir problemi çözmek için izlenen adım adım yöntemdir.
İşlemci, bilgisayarda komutları çalıştıran ana bileşendir.
Bellek, çalışan programların geçici verilerini hızlıca saklar.
Depolama birimleri, dosyaları kapatıldıktan sonra da saklar.
İşletim sistemi, donanımı yönetir ve programların çalışmasını sağlar.
İnternet, dünyadaki bilgisayar ağlarını birbirine bağlayan küresel bir ağdır.
Bir web tarayıcısı, internet sayfalarını görüntülemeye yarayan programdır.
E-posta, elektronik ortamda mesaj göndermeye yarayan bir iletişim yöntemidir.
Güçlü bir parola uzun olmalı ve başka hesaplarda kullanılmamalıdır.
İki adımlı doğrulama, hesabı yalnızca parolaya bağlı olmaktan kurtarır.
Yedekleme, önemli dosyaların bir kopyasını güvenli bir yerde saklamaktır.
Şifreleme, veriyi anahtarı olmayanların okuyamayacağı hale getirir.
Kimlik avı mesajları, kullanıcıyı sahte bir sayfada bilgi girmeye yönlendirmeye çalışır.
Yazılım güncellemeleri, güvenlik açıklarını kapatmaya yardımcı olur.
Bir değişken, programda değer saklayan adlandırılmış alandır.
Döngü, belirli komutların tekrar tekrar çalışmasını sağlar.
Fonksiyon, tekrar kullanılabilen ve adı olan bir kod parçasıdır.
Hata ayıklama, programdaki hatanın nedenini bulup düzeltme sürecidir.
Test yazmak, kodun beklenen sonucu verdiğini otomatik olarak denetler.
Sürüm kontrol sistemi, dosyalardaki değişikliklerin geçmişini tutar.
Veritabanı, düzenli biçimde saklanan ve sorgulanabilen veri koleksiyonudur.
Bir API, iki yazılımın birbiriyle konuşmasını sağlayan kurallar bütünüdür.
Bulut bilişim, sunucuların ve depolamanın internet üzerinden kullanılmasıdır.
Yapay zekâ, insan zekâsını gerektiren bazı işleri yapabilen sistemlerin genel adıdır.
Makine öğrenmesi, bir sistemin örneklerden kalıp öğrenmesine dayanır.
Bir dil modeli, metindeki sonraki parçayı tahmin etmeyi öğrenerek eğitilir.
Eğitim verisinin kalitesi, bir modelin davranışını doğrudan etkiler.
Bir model, eğitim verisinde olmayan konularda yanılabilir.
Doğrulama kümesi, eğitim sırasında modelin gelişimini izlemek için ayrılır.
Test kümesi, eğitim bittikten sonra modelin başarısını tarafsızca ölçmek içindir.
Aşırı öğrenme, modelin örnekleri ezberleyip yeni verilerde başarısız olmasıdır.
Kayıp değeri, modelin tahminlerinin doğru cevaptan ne kadar uzak olduğunu gösterir.
Bir belirteç, metnin modelin işlediği küçük parçalarından biridir.
Ekran okuyucular, görme engelli kullanıcıların sayfaları dinlemesine yardımcı olur.
Bir pikselin rengi kırmızı, yeşil ve mavi değerlerinin birleşimiyle belirlenir.
Ses dosyaları, havadaki titreşimlerin sayısal örneklerle kaydedilmesiyle oluşur.
Bir görüntü, satır ve sütunlara dizilmiş piksellerden oluşur.
Pil, kimyasal enerjiyi elektrik enerjisine çeviren bir kaynaktır.
Sensörler, ortamdaki ısı, ışık veya hareket gibi değişimleri ölçer.
Robotlar, algıladıkları bilgiye göre hareket edebilen makinelerdir.
Üç boyutlu bir model, uzayda nokta ve yüzeylerle tanımlanır.
Bir belge dosyası metni, başlıkları ve tabloları birlikte saklayabilir.
Dosya adlarında anlaşılır ve tutarlı isimler kullanmak, sonradan bulmayı kolaylaştırır.
Ağ bağlantısı koptuğunda, çevrimiçi hizmetler geçici olarak çalışmayabilir.
Gizlilik ayarları, hangi bilgilerin kimlerle paylaşılacağını belirler.
Açık kaynak yazılımda kaynak kod herkesin incelemesine açıktır.
Bir lisans, yazılımın nasıl kullanılabileceğini belirleyen kurallardır.
Kullanıcı arayüzü, kişinin bir programla etkileşim kurduğu görünür kısımdır.""",
"gunluk": """Düzenli uyku, dikkat ve ruh hali için önemlidir.
Gün içinde yeterli su içmek vücudun çalışmasına yardımcı olur.
Dengeli beslenme, sebze, meyve, tahıl ve proteinin birlikte tüketilmesini içerir.
Yürüyüş, özel bir ekipman gerektirmeyen kolay bir egzersizdir.
Ellerin sabunla yıkanması, mikropların yayılmasını azaltır.
Bir bütçe hazırlamak, gelir ve giderleri birlikte görmenin yoludur.
Acil bir ihtiyaç için küçük bir birikim ayırmak sıkıntıyı azaltır.
Alışveriş listesi hazırlamak, gereksiz harcamaları azaltabilir.
Bir toplantıya zamanında gelmek, diğer insanlara saygının bir göstergesidir.
Yeni bir şehre giderken haritayı önceden incelemek yol bulmayı kolaylaştırır.
Toplu taşıma, kalabalık şehirlerde trafiği azaltmaya yardımcı olur.
Bisiklet, kısa mesafeler için sağlıklı ve çevre dostu bir ulaşım aracıdır.
Mutfakta çiğ et ve sebze için ayrı tahtalar kullanmak hijyen açısından yararlıdır.
Çorba, sebze ve baklagillerle kolayca hazırlanabilen sıcak bir yemektir.
Çay demlerken suyun taze olması tadı etkiler.
Ekmek, un, su, tuz ve mayadan yapılan temel bir gıdadır.
Bir tarifi uygularken ölçüleri doğru kullanmak sonucu iyileştirir.
Evde bitki yetiştirmek için ışık, su ve uygun toprak gerekir.
Balkonda küçük bir bahçe yapmak, şehirde yaşayanlara doğayla bağ kurma fırsatı verir.
Elektrikli aletleri kullanmadan önce kablolarını kontrol etmek güvenlik sağlar.
Yangın anında asansör yerine merdivenin kullanılması önerilir.
Önemli telefon numaralarını kolay ulaşılır bir yerde saklamak acil durumda işe yarar.
Bir ilaç kullanmadan önce prospektüsü okumak ve gerekirse doktora danışmak önemlidir.
Gözleri uzun süre ekrana bakarken dinlendirmek yorgunluğu azaltır.
Mola vermek, uzun çalışma saatlerinde verimi korumaya yardım eder.
Bir işi küçük adımlara bölmek, zor görünen hedefleri yönetilebilir yapar.
Yapılacaklar listesi, günün önceliklerini görmeyi kolaylaştırır.
Okumak, kelime dağarcığını ve düşünme becerisini geliştirir.
Yeni bir dil öğrenirken her gün kısa süre çalışmak uzun ve seyrek çalışmadan genellikle daha etkilidir.
Not almak, duyulan bilgiyi daha iyi hatırlamaya yardımcı olabilir.
Bir sunum hazırlarken tek slayta çok fazla yazı koymamak izleyicinin takibini kolaylaştırır.
Mektup yazarken hitap ve kapanış cümlesi, metne nezaket katar.
Komşularla iyi ilişki kurmak, apartman yaşamını kolaylaştırır.
Bir hediye seçerken kişinin ilgi alanlarını düşünmek, hediyeyi anlamlı kılar.
Doğum günü kutlamak, bir insana değer verdiğini göstermenin yollarından biridir.
Bayramlarda akrabaları ziyaret etmek ve hâl hatır sormak bir gelenektir.
Misafir ağırlarken sunulan ikram, ev sahibinin nezaketini yansıtır.
Yolculukta bagajı önceden hazırlamak, son anda telaşı azaltır.
Pasaport ve kimlik gibi belgelerin kopyasını ayrı yerde saklamak, kaybolma durumunda kolaylık sağlar.
Mevsime uygun giyinmek, hava değişimlerinde hasta olma riskini azaltabilir.
Şemsiye, yağmurlu günlerde yanımızda bulundurulması gereken pratik bir eşyadır.
Evi düzenli tutmak, aranan şeyi bulmayı kolaylaştırır.
Atık pili çöpe atmak yerine toplama kutusuna bırakmak doğayı korur.
Kağıt, cam ve plastik atıkları ayırmak geri dönüşüme katkı sağlar.
Su tasarrufu için musluğu gereksiz yere açık bırakmamak önemlidir.
Işıkları kullanılmayan odalarda kapatmak elektrik tasarrufu sağlar.
Hayvanlara iyi davranmak, toplumun merhamet duygusunu gösterir.
Gönüllü çalışmalar, insanların birbirine destek olmasını sağlar.
Yaşlılara yardım etmek ve onlara saygı göstermek toplumsal bir değerdir.
Bir sorunu çözerken sakin kalmak doğru karar vermeye yardımcı olur.
Hata yapmak, öğrenme sürecinin doğal bir parçasıdır.""",
"dil": """Türkçede ekler, kelimenin sonuna eklenerek anlamı ve görevi değiştirir.
Türkçe cümlede yüklem çoğunlukla sonda yer alır.
Büyük ünlü uyumu, bir kelimedeki ünlülerin kalın veya ince olmasına dayanan bir kuraldır.
Noktalama işaretleri, yazıda duraklama ve anlam ayrımını göstermeye yarar.
Soru cümlelerinde "mi" eki ayrı yazılır ve önceki kelimeye anlam katar.
Cümle büyük harfle başlar ve nokta, soru veya ünlem işaretiyle biter.
Özel isimler büyük harfle başlar.
Eş anlamlı kelimeler, aynı ya da birbirine çok yakın anlamları taşır.
Zıt anlamlı kelimeler, birbirine karşıt anlamlar ifade eder.
Atasözleri, uzun deneyimlerden doğan kısa ve öğüt veren sözlerdir.
Deyimler, kelimelerin gerçek anlamlarından farklı bir anlam kazandığı kalıplardır.
Bir paragraf, tek bir ana düşünce etrafında toplanan cümleler topluluğudur.
Giriş, gelişme ve sonuç, bir yazının temel bölümleridir.
Başlık, okuyucuya metnin konusu hakkında fikir verir.
Özet, bir metnin ana fikrini daha kısa biçimde aktarmaktır.
Bir e-postada konu satırının açık olması, alıcının mesajı anlamasını kolaylaştırır.
Resmi bir yazışmada saygılı ve sade bir dil kullanmak tercih edilir.
Arkadaşlar arasında konuşurken daha rahat bir dil kullanılabilir, ancak başkasını incitmemeye dikkat edilmelidir.
Nazik bir dil, karşıdakine değer verildiğini gösterir.
Sinirli olunduğunda bile kırıcı sözlerden kaçınmak ilişkiyi korur.
Hakaret içeren sözler, insanların kalbini kırar ve sorunu çözmez.
Bir görüşe katılmadığımızı söylerken saygılı ve açık bir üslup seçmek daha yapıcıdır.
Özür dilemek, hatayı kabul etmenin ve ilişkiyi onarmanın bir yoludur.
Teşekkür etmek, yapılan iyiliğin fark edildiğini gösterir.
Lütfen kelimesi, rica ederken kullanılan kibar bir ifadedir.
Dinlemek, iyi iletişimin konuşmak kadar önemli bir parçasıdır.
Soru sormak, anlamadığımız bir konuyu öğrenmenin en doğal yoludur.
Bir tanımı verirken önce genel sınıfı, sonra ayırt edici özelliği söylemek anlaşılırlığı artırır.
Örnek vermek, soyut bir açıklamayı somut hale getirir.
Bir karşılaştırmada benzerlikler ve farklar ayrı ayrı belirtilirse okuyucu daha rahat anlar.
Kısa cümleler, karmaşık fikirleri açık anlatmaya yardımcı olur.
Bir metni yazdıktan sonra yeniden okumak, yazım hatalarını yakalamayı sağlar.
Kaynak belirtmek, bilginin nereden geldiğini okuyucunun denetleyebilmesini sağlar.
Emin olunmayan bir bilgiyi kesinmiş gibi söylememek dürüst iletişimin gereğidir.
Bilmediğimiz bir şeyi bilmediğimizi söylemek, güveni artırır.
Eleştiri yaparken kişiyi değil davranışı ele almak daha yapıcıdır.
Bir talimat verirken adımları sırayla ve kısa yazmak anlaşılmayı kolaylaştırır.
Madde işaretleri, birbirinden ayrı fikirleri düzenli göstermeye yarar.
Tablolar, karşılaştırılacak bilgileri satır ve sütunlarda toplar.
Rakamlarla yazılan tarihlerde gün, ay ve yıl sırasına dikkat etmek karışıklığı önler.
Ondalık sayılarda Türkçede virgül kullanılır.
Saat bilgisi verirken gün içindeki zaman diliminin belirtilmesi yanlış anlamayı önler.
Bir toplantı notunda kararlar ve sorumlular açıkça yazılmalıdır.
Yazılı bir metinde ton, okuyucunun mesajı nasıl algılayacağını etkiler.
Tebrik ve taziye mesajları kısa, samimi ve saygılı olmalıdır.
Çocuklara anlatırken basit sözcükler ve somut örnekler kullanmak anlamayı kolaylaştırır.
Yabancı bir kelimenin Türkçe karşılığı varsa kullanmak dili zenginleştirir.
Dil, toplumların ortak düşünce ve duygularını aktaran canlı bir sistemdir.
Bir hikâyede olay, kişi, yer ve zaman unsurları bulunur.
Şiirde dizeler ve uyak, ritim duygusu yaratır.""",
}

def paragraphs(lines, size=3):
    if len(lines) < size:
        return [" ".join(lines)]
    return [" ".join(lines[i:i + size]) for i in range(0, len(lines) - size + 1, 1)]


def chunks(items, n):
    out = [items[i:i + n] for i in range(0, len(items), n)]
    if len(out) > 1 and len(out[-1]) < 4:
        last = out.pop()
        out[-1] = out[-1] + last
    return out


def write(name, recs):
    (HERE / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")


def main():
    import importlib.util
    for old in HERE.glob("*.jsonl"):
        old.unlink()
    spec = importlib.util.spec_from_file_location("make_expand", HERE.parent / "make_expand.py")
    me = importlib.util.module_from_spec(spec); spec.loader.exec_module(me)
    groups = {k: [l.strip() for l in v.strip().splitlines() if l.strip()] for k, v in GROUPS.items()}
    groups.update(me.extra_groups())
    total = 0
    for name, lines in groups.items():
        for n, part in enumerate(chunks(lines, 12), 1):
            recs = [{"text": p} for p in paragraphs(part)]
            write(f"{name}-{n:02d}", recs); total += len(recs)
    chat = [json.loads(l) for l in (HERE.parent / "chat" / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for n, part in enumerate(chunks(chat, 25), 1):
        recs = [{"text": f"Kullanıcı: {c['prompt']}\nAETHRION: {c['response']}"} for c in part]
        write(f"diyalog-{n:02d}", recs); total += len(recs)
    print(len(list(HERE.glob('*.jsonl'))), "source groups,", total, "records")


if __name__ == "__main__":
    main()
