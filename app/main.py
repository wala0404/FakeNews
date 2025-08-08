from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import classify, ocr, recommend

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/api")
app.include_router(ocr.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
