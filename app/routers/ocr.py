from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import OCRResponse
from app.ml.infer import ocr_image

router = APIRouter()

@router.post("/ocr", response_model=OCRResponse)
def ocr(file: UploadFile = File(...)):
    try:
        text = ocr_image(file)
        return OCRResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

