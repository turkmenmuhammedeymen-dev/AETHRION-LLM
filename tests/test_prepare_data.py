import json
import tempfile
import unittest
from pathlib import Path
from aethrion.prepare_data import build_corpus, load_sources_manifest, normalize_text

class CorpusPreparationTests(unittest.TestCase):
    def test_unicode_and_whitespace(self):
        self.assertEqual(normalize_text("  Türkçe\n  metin  "), "Türkçe metin")

    def test_manifest_pipeline_deduplicates_and_filters_email(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"source.txt").write_text(
              ("Bu, AETHRION için yeterince uzun bir Türkçe eğitim cümlesidir.\n")*2+
              "İkinci ve farklı bir Türkçe eğitim örneği burada bulunuyor.\n"+
              "İletişim test@example.com olan kayıt dışarıda kalmalıdır.\n",encoding="utf-8")
            manifest=root/"sources.json"
            manifest.write_text(json.dumps({"sources":[{"source_id":"unit-test","path":"source.txt",
              "license":"test","permitted_uses":"unit test","allow_training":True}]}),encoding="utf-8")
            output=root/"out/corpus.txt"; report_path=root/"out/report.json"
            report=build_corpus(root,manifest,output,report_path,seed=7,min_chars=10,min_records=1)
            self.assertEqual(report["totals"]["records_kept"],2)
            self.assertEqual(report["totals"]["duplicates"],1)
            self.assertEqual(report["totals"]["email_like_records_excluded"],1)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()),2)
            self.assertTrue(report_path.exists())

    def test_rejects_unapproved_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/"source.txt").write_text("some text",encoding="utf-8")
            manifest=root/"sources.json"
            manifest.write_text(json.dumps({"sources":[{"source_id":"blocked","path":"source.txt",
              "license":"unknown","permitted_uses":"none","allow_training":False}]}),encoding="utf-8")
            with self.assertRaises(ValueError): load_sources_manifest(manifest,root)

if __name__ == "__main__": unittest.main()
