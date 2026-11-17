import base64, io, json, tempfile, unittest, zipfile
from pathlib import Path
from aethrion.audio import OpenAISpeechProvider, AudioProviderError
from aethrion.document_tools import create_document, extract_document_text, DocumentError
from aethrion.workspace_agent import read_project_zip, WorkspaceError

class ExtendedMultimodalTests(unittest.TestCase):
    def test_audio_fails_closed_without_key(self):
        provider=OpenAISpeechProvider(api_key="")
        with self.assertRaises(AudioProviderError):
            provider.synthesize("Hello")

    def test_document_creation_and_text_roundtrip(self):
        raw,name,mime=create_document("AETHRION","Merhaba dünya","txt")
        self.assertTrue(name.endswith(".txt"))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/name; path.write_bytes(raw)
            self.assertIn("Merhaba dünya",extract_document_text(path))

    def test_document_rejects_oversized_title(self):
        with self.assertRaises(DocumentError):
            create_document("x"*181,"content")

    def test_workspace_rejects_path_traversal(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z: z.writestr("../escape.py","print(1)")
        with self.assertRaises(WorkspaceError):
            read_project_zip(buf.getvalue())

    def test_workspace_reads_small_python_project(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z: z.writestr("main.py","print('hello')\n")
        self.assertEqual(read_project_zip(buf.getvalue()),{"main.py":"print('hello')\n"})

if __name__=="__main__": unittest.main()
