"""
routes/predict.py - Prediction endpoints
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel

from services.predictor import run_prediction
from services.report_generator import generate_pdf_report

router = APIRouter(tags=["Prediction"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class PredictionResponse(BaseModel):
    tumor_detected: bool
    tumor_type: Optional[str]
    confidence: float
    uncertainty: float
    entropy: float
    reliability: str
    risk_level: str
    risk_color: str
    clinical_note: str
    recommendation: str
    heatmap_image: str
    scorecam_image: str
    comparison_strip: str
    all_class_probs: Dict[str, float]
    tta_agreement: float
    localization: Optional[Dict[str, Any]] = None


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Upload an MRI image and receive AI analysis",
)
async def predict(
    file: UploadFile = File(..., description="MRI image (JPEG / PNG / BMP / TIFF)"),
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: JPEG, PNG, BMP, TIFF.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10 MB limit.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    result = await run_prediction(image_bytes)

    return PredictionResponse(
        tumor_detected=result.tumor_detected,
        tumor_type=result.tumor_type,
        confidence=result.confidence,
        uncertainty=result.uncertainty,
        entropy=result.entropy,
        reliability=result.reliability,
        risk_level=result.risk_level,
        risk_color=result.risk_color,
        clinical_note=result.clinical_note,
        recommendation=result.recommendation,
        heatmap_image=result.heatmap_image,
        scorecam_image=result.scorecam_image,
        comparison_strip=result.comparison_strip,
        all_class_probs=result.all_class_probs,
        tta_agreement=result.tta_agreement,
        localization=result.localization,
    )


@router.post("/predict/report", summary="Generate a clinical PDF report")
async def get_report(prediction_data: dict):
    try:
        pdf_bytes = generate_pdf_report("Anonymous", prediction_data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=clinical_report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
