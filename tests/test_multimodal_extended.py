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

    def test_native_vision_rejects_oversized_image_dimensions_before_model_load(self):
        from unittest.mock import patch
        from aethrion.native_runtime import NativeModelRuntime, NativeRuntimeError

        class OversizedImage:
            format = "PNG"
            size = (9000, 100)
            def __enter__(self): return self
            def __exit__(self, *args): return False

        runtime = NativeModelRuntime.__new__(NativeModelRuntime)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.png"
            path.write_bytes(b"not-decoded-test-image")
            with patch("PIL.Image.open", return_value=OversizedImage()):
                with self.assertRaisesRegex(NativeRuntimeError, "safe dimensions"):
                    runtime.image_text_similarity(path, "description")

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

    def test_workspace_model_changes_reject_duplicate_normalized_paths(self):
        from aethrion.workspace_agent import _validate
        with self.assertRaisesRegex(WorkspaceError, "Duplicate normalized"):
            _validate({"folder\\\\" + "main.py": "print(1)", "folder/main.py": "print(2)"}, {})

    def test_workspace_model_changes_enforce_total_updated_text_budget(self):
        from aethrion.workspace_agent import MAX_PROJECT_TEXT, _validate
        with self.assertRaisesRegex(WorkspaceError, "total text budget"):
            _validate({"large.py": "x" * (MAX_PROJECT_TEXT + 1)}, {})

    def test_workspace_rejects_non_string_archive_paths(self):
        from aethrion.workspace_agent import _safe_name
        with self.assertRaises(WorkspaceError):
            _safe_name(123)

    def test_workspace_reads_small_python_project(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z: z.writestr("main.py","print('hello')\n")
        self.assertEqual(read_project_zip(buf.getvalue()),{"main.py":"print('hello')\n"})

if __name__=="__main__": unittest.main()
