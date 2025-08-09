from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
import io

# This line is crucial - must create the router instance
router = APIRouter()

@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    text = pytesseract.image_to_string(image, lang='ara')
    return {"text": text}

