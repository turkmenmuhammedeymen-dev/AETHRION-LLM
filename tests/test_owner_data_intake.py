import json, tempfile, unittest
from pathlib import Path
from scripts.create_owner_data_intake import SPECS, create_intake_workspace

class OwnerDataIntakeTests(unittest.TestCase):
 def test_creates_eleven_empty_fail_closed_intake_folders(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/"intake"
   report=create_intake_workspace(root)
   self.assertEqual(report["model_count"],11)
   self.assertFalse(report["training_started"])
   self.assertFalse(report["automatic_approval"])
   self.assertEqual(set(SPECS),{x["model_id"] for x in report["models"]})
   for model,(_,_,fields,assets,_) in SPECS.items():
    folder=root/model
    self.assertEqual((folder/"records.jsonl").read_text(encoding="utf-8"),"")
    submission=json.loads((folder/"submission.json").read_text(encoding="utf-8"))
    self.assertIs(submission["training_allowed"],False)
    self.assertIs(submission["owner_approval"],False)
    self.assertEqual(submission["source_count"],0)
    self.assertEqual(submission["record_fields"],fields)
    governance=json.loads((folder/"governance.json").read_text(encoding="utf-8"))
    self.assertEqual(governance["approval_status"],"pending")
    for key in ("authorship_attested","first_party_origin_attested","rights_attested","source_rights_reviewed","privacy_reviewed","quality_reviewed"):
     self.assertIs(governance[key],False)
    if model in {"asr","tts","vocoder"}: self.assertIs(governance["speaker_consent_reviewed"],False)
    if model=="safety": self.assertIs(governance["safety_reviewed"],False)
    for asset in assets: self.assertTrue((folder/asset).is_dir())
 def test_refuses_to_overwrite_existing_workspace(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/"intake";root.mkdir()
   sentinel=root/"keep.txt";sentinel.write_text("preserve")
   with self.assertRaises(FileExistsError): create_intake_workspace(root)
   self.assertEqual(sentinel.read_text(),"preserve")
 def test_exact_eleven_model_inventory(self):
  self.assertEqual(set(SPECS),{"chat","code","vision","image","asr","tts","vocoder","3d","research","document","safety"})

if __name__=="__main__": unittest.main()
