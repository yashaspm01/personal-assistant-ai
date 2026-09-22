"""OCR text extraction for uploaded transaction screenshots (bank/UPI app
screenshots) — falls back point for images, distinct from doc_loader.py
which handles real text-based documents (PDF/DOCX/TXT)."""
import io
from PIL import Image
import pytesseract

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


def is_image_file(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS)
