"""
main.py - FastAPI application entry point
Run with:
  uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Brain Tumor Analysis API starting up...")
    print("System ready.")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Brain Tumor Analysis API",
    description=(
        "Explainable brain tumor detection and classification with uncertainty "
        "estimation and PDF report generation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "Brain Tumor Analysis API",
        "version": "2.0.0",
        "docs": "/docs",
        "features": ["prediction", "pdf-reports", "explainability", "uncertainty"],
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
