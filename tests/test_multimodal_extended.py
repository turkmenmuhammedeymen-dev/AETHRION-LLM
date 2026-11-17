import base64, io, json, tempfile, unittest, zipfile
from pathlib import Path
from aethrion.document_tools import create_document, extract_document_text, DocumentError
from aethrion.workspace_agent import read_project_zip, WorkspaceError

class ExtendedMultimodalTests(unittest.TestCase):
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

    def test_workspace_rejects_duplicate_paths(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr("main.py","print('one')\\n")
            z.writestr("main.py","print('two')\\n")
        with self.assertRaisesRegex(WorkspaceError,"Duplicate project path"):
            read_project_zip(buf.getvalue())

    def test_workspace_rejects_decompressed_entry_over_limit(self):
        from aethrion.workspace_agent import MAX_FILE_BYTES
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("large.py","x"*(MAX_FILE_BYTES+1))
        with self.assertRaisesRegex(WorkspaceError,"per-file size limit"):
            read_project_zip(buf.getvalue())

    def test_docx_archive_rejects_missing_required_document_xml(self):
        from aethrion.document_tools import _validate_docx_archive
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr("word/styles.xml","<styles/>")
        with self.assertRaisesRegex(DocumentError,"missing word/document.xml"):
            _validate_docx_archive(buf.getvalue())

    def test_docx_archive_rejects_excessive_uncompressed_entry(self):
        from aethrion.document_tools import _validate_docx_archive
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml","x"*(16*1024*1024+1))
        with self.assertRaisesRegex(DocumentError,"oversized entry"):
            _validate_docx_archive(buf.getvalue())

    def test_workspace_reads_small_python_project(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z: z.writestr("main.py","print('hello')\n")
        self.assertEqual(read_project_zip(buf.getvalue()),{"main.py":"print('hello')\n"})

if __name__=="__main__": unittest.main()
