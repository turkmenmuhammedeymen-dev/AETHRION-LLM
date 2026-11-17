"""Bounded document extraction, generation and text-first editing."""
from __future__ import annotations
import csv, io, json
from pathlib import Path
from typing import Any

class DocumentError(ValueError):
    pass

MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 200_000

def extract_document_text(path):
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_INPUT_BYTES:
        raise DocumentError("File is missing or larger than 8 MiB.")
    suffix = source.suffix.lower()
    raw = source.read_bytes()
    if suffix in {".txt",".md",".log",".json",".csv"}:
        text = raw.decode("utf-8-sig", errors="strict")
        if suffix == ".json":
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        elif suffix == ".csv":
            text = "\\n".join(" | ".join(row) for row in list(csv.reader(io.StringIO(text)))[:5000])
        return text[:MAX_TEXT_CHARS]
    if suffix == ".docx":
        try:
            from docx import Document
            document = Document(io.BytesIO(raw))
            parts = [p.text for p in document.paragraphs if p.text]
            for table in document.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            return "\\n".join(parts)[:MAX_TEXT_CHARS]
        except Exception as exc:
            raise DocumentError("DOCX could not be read.") from exc
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\\n".join(page.extract_text() or "" for page in reader.pages[:100])[:MAX_TEXT_CHARS]
        except Exception as exc:
            raise DocumentError("PDF could not be read.") from exc
    raise DocumentError("Supported formats: TXT, MD, LOG, JSON, CSV, DOCX and PDF.")

def create_document(title, content, output_format="docx"):
    if not isinstance(title, str) or not title.strip() or len(title) > 180:
        raise DocumentError("Title must contain 1–180 characters.")
    if not isinstance(content, str) or not content.strip() or len(content) > MAX_TEXT_CHARS:
        raise DocumentError("Content must contain 1–200,000 characters.")
    fmt = output_format.lower().lstrip(".")
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:80] or "document"
    if fmt in {"txt","md"}:
        body = (f"# {title.strip()}\\n\\n" if fmt == "md" else f"{title.strip()}\\n\\n") + content
        return body.encode("utf-8"), f"{safe_title}.{fmt}", "text/markdown" if fmt == "md" else "text/plain"
    if fmt == "json":
        return json.dumps({"title":title.strip(),"content":content},ensure_ascii=False,indent=2).encode(),f"{safe_title}.json","application/json"
    if fmt == "csv":
        out=io.StringIO(); writer=csv.writer(out); writer.writerow(["title","content"]); writer.writerow([title.strip(),content])
        return out.getvalue().encode(),f"{safe_title}.csv","text/csv"
    if fmt == "docx":
        try:
            from docx import Document
            doc=Document(); doc.add_heading(title.strip(),level=0)
            for block in content.split("\\n\\n"): doc.add_paragraph(block.strip())
            out=io.BytesIO(); doc.save(out)
            return out.getvalue(),f"{safe_title}.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        except Exception as exc:
            raise DocumentError("DOCX writer unavailable.") from exc
    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
            from xml.sax.saxutils import escape
            out=io.BytesIO(); pdf=SimpleDocTemplate(out,pagesize=A4,rightMargin=48,leftMargin=48,topMargin=48,bottomMargin=48)
            styles=getSampleStyleSheet(); story=[Paragraph(escape(title.strip()),styles["Title"]),Spacer(1,16)]
            for block in content.split("\\n\\n"):
                story.extend([Paragraph(escape(block).replace("\\n","<br/>"),styles["BodyText"]),Spacer(1,10)])
            pdf.build(story)
            return out.getvalue(),f"{safe_title}.pdf","application/pdf"
        except Exception as exc:
            raise DocumentError("PDF writer unavailable.") from exc
    raise DocumentError("Output format must be DOCX, PDF, TXT, MD, JSON or CSV.")

def replace_document_text(source_path, find_text, replace_text, output_format="docx"):
    if not find_text:
        raise DocumentError("Find text cannot be empty.")
    content=extract_document_text(source_path)
    if find_text not in content:
        raise DocumentError("Requested text was not found.")
    return create_document(Path(source_path).stem+"-edited",content.replace(find_text,replace_text),output_format)
