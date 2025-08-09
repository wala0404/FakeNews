from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ocr, classify, recommend  # Make sure all routers are imported

app = FastAPI(title="FakeNews Detector", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "FakeNews Detection API",
        "endpoints": {
            "docs": "/docs",
            "ocr": "/api/ocr",
            "classify": "/api/classify",
            "recommend": "/api/recommend"
        }
    }

# Include all routers
app.include_router(ocr.router, prefix="/api")
app.include_router(classify.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")