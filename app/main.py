from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ocr, classify, recommend  # include recommend router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include both routers
app.include_router(ocr.router, prefix="/api")
app.include_router(classify.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "News Verification API is running"}