"""Extracts raw text from uploaded PDF, DOCX, or TXT files.

For PDFs: tries normal text extraction first (fast, works for most files).
If that yields nothing — common for scanned documents or PDFs exported from
design tools as outlined graphics rather than encoded text — falls back to
rendering each page as an image and running OCR on it, reusing the same
Tesseract pipeline already built for Transactions screenshots.
"""
import io
from pypdf import PdfReader
from docx import Document
import fitz  # PyMuPDF
from app.services.ocr_loader import extract_text_from_image


def _extract_pdf_text_native(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_pdf_text_via_ocr(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_texts = []
    for page in doc:
        # 200 DPI-equivalent render — good balance of OCR accuracy vs speed
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pix.tobytes("png")
        page_texts.append(extract_text_from_image(image_bytes))
    doc.close()
    return "\n".join(page_texts)


def extract_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".pdf"):
        text = _extract_pdf_text_native(file_bytes)
        if text.strip():
            return text
        # No text layer found — likely scanned/image-based, fall back to OCR
        return _extract_pdf_text_via_ocr(file_bytes)

    elif lower.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    elif lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {filename}")
