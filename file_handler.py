import re
import os
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
import subprocess
import logging

logger = logging.getLogger(__name__)

def extract_text(file_path: str) -> str:
    """Return cleaned plain text from PDF, DOCX, DOC, TXT."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif ext == ".doc":
            result = subprocess.run(
                ["antiword", str(file_path)], capture_output=True, text=True
            )
            text = result.stdout if result.returncode == 0 else ""
        elif ext == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            return ""
    except Exception as e:
        logger.error(f"Cannot extract text from {file_path}: {e}")
        return ""

    # basic cleanup
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()