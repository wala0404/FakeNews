from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import pytesseract
import io

router = APIRouter()

@router.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form(...)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))


        image = ImageOps.grayscale(image)

        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config='--oem 1 --psm 6'
        )

        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
