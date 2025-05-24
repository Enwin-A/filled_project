# from typing import Optional

# from fastapi import FastAPI, File, UploadFile

# app = FastAPI()


# @app.post("/classify")
# async def schedule_classify_task(file: Optional[UploadFile] = File(None)):
#     """Endpoint to classify a document into "w2", "1099int", etc"""

#     return {"document_type": "NOT IMPLEMENTED", "year": "NOT IMPLEMENTED"}


from typing import Optional
import re
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
import fitz 
from PIL import Image
import pytesseract 
from pytesseract import TesseractNotFoundError
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = FastAPI()


CARD_MODEL = YOLO("./best.pt")
FORM_DETECTION_ORDER = ["W2", "1040", "1099INT", "1099DIV"]
FORM_PATTERNS = {
    "W2": [
        r"(?i)Form\s*W-2.*Wage\s*and\s*Tax\s*Statement.*\d{4}",
        r"(?i)Employer['’]s\s+(name|ID).*Employee['’]s\s+(social security|name)"
    ],
    "1040": [
        r"(?m)^Form\s*1040\b.*U\.S\. Individual Income Tax Return",
        r"(?i)Adjusted\s+Gross\s+Income.*Form\s*1040"
    ],
    "1099INT": [r"\bForm\s*1099-INT\b", r"\b1099-INT\b"],
    "1099DIV": [r"\bForm\s*1099-DIV\b", r"\b1099-DIV\b"],
}


DATE_REGEX = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b")
YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")

ID_CARD     = "ID_CARD"
HANDWRITTEN = "HANDWRITTEN"
OTHER       = "OTHER"



def extract_date(text: str) -> str:
    m = DATE_REGEX.search(text or "")
    if m:
        return m.group(0)
    m2 = YEAR_REGEX.search(text or "")
    return m2.group(0) if m2 else "UNKNOWN"


def detect_form(text: str) -> Optional[str]:
    for form in FORM_DETECTION_ORDER:
        for pattern in FORM_PATTERNS[form]:
            if re.search(pattern, text, re.IGNORECASE):
                return form
    return None

def perform_ocr(images):
    try:
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except TesseractNotFoundError:
        print("Tesseract not found! Verify installation at C:\Program Files\Tesseract-OCR")
        return ""
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

@app.post("/classify")
async def classify_document(file: Optional[UploadFile] = File(None)):
    if not file or file.content_type != "application/pdf":
        raise HTTPException(400, "A PDF file is required.")

    data = await file.read()

    reader = PdfReader(io.BytesIO(data))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text).strip()

    if full_text:
        form = detect_form(full_text)
        if form:
            return {"document_type": form, "year": extract_date(full_text)}

    pdf_doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for page in pdf_doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    for img in images:
        results = CARD_MODEL.predict(img, conf=0.25)
        if any(len(r.boxes) > 0 for r in results):
            ocr_txt = perform_ocr(images)
            return {"document_type": ID_CARD, "year": extract_date(ocr_txt or full_text)}

    text_source = full_text or perform_ocr(images)
    doc_type = OTHER if text_source.strip() else HANDWRITTEN
    return {"document_type": doc_type, "year": extract_date(text_source)}