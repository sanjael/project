"""
services/predictor.py  (UPGRADED v2)
──────────────────────────────────────
Advanced AI inference pipeline:

1. CLAHE + skull-strip preprocessing (cleaner input)
2. Test-Time Augmentation (TTA) — 3 augmented views, averaged predictions
3. MC Dropout Bayesian uncertainty — 10 forward passes
4. Temperature scaling for calibrated probabilities
5. Dual explainability: Grad-CAM++ + Score-CAM (more faithful localisation)
6. Entropy-based uncertainty (captures distributional uncertainty)
7. Confidence-weighted clinical risk integration
8. Fallback: EfficientNet-B4 if available; ResNet101 otherwise
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from config import get_settings
from services.preprocessing import (
    load_image_from_bytes,
    preprocess_for_inference,
    preprocess_tta,
)
from services.model_loader import get_detection_model, get_classification_model, get_device
from services.gradcam import GradCAMPlusPlus, ScoreCAM, generate_comparison_strip, localize_cam
from services.risk_analysis import get_risk_report
from models.resnet_models import TUMOR_CLASSES

settings = get_settings()


# ── Temperature scaling for calibrated probabilities ──────────────────────────

_TEMPERATURE = 1.3  # > 1 softens overconfident predictions


def _apply_temperature(logits: torch.Tensor, binary: bool = False) -> torch.Tensor:
    """Scale logits by temperature before sigmoid/softmax."""
    scaled = logits / _TEMPERATURE
    if binary:
        return torch.sigmoid(scaled)
    return torch.softmax(scaled, dim=-1)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    tumor_detected: bool
    tumor_type: Optional[str]
    confidence: float               # 0.0 – 1.0
    uncertainty: float              # MC Dropout std dev
    entropy: float                  # Shannon entropy of class probs
    reliability: str
    risk_level: str
    risk_color: str
    clinical_note: str
    recommendation: str
    heatmap_image: str              # Grad-CAM++ overlay base64
    scorecam_image: str             # Score-CAM overlay base64
    comparison_strip: str           # Side-by-side strip base64
    all_class_probs: dict           # {class: prob}
    tta_agreement: float            # 0–1: how consistent TTA views are
    localization: Optional[dict]    # bbox + area_pct from CAM
    calibrated: bool = True


# ── MC Dropout activator ──────────────────────────────────────────────────────

def _enable_mc_dropout(model: torch.nn.Module):
    """Keep Dropout layers in train() mode for stochastic inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


# ── Shannon entropy ───────────────────────────────────────────────────────────

def _entropy(probs: torch.Tensor) -> float:
    """
    Compute normalised Shannon entropy H(p) / log(N).
    Returns 0.0 for perfectly certain predictions, 1.0 for maximum uncertainty.
    """
    probs = probs.clamp(min=1e-8)
    h = -(probs * probs.log()).sum(dim=-1).mean().item()
    n = probs.shape[-1]
    return round(h / np.log(n), 4) if n > 1 else round(h / np.log(2), 4)


# ── TTA Prediction ────────────────────────────────────────────────────────────

def _tta_predict_detection(model, tta_tensors: List[torch.Tensor], device: str) -> List[float]:
    """Returns a prob per TTA view (list of floats)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for t in tta_tensors:
            logit = model(t.to(device)).squeeze()
            p = torch.sigmoid(logit / _TEMPERATURE).item()
            probs.append(p)
    return probs


def _tta_predict_classification(model, tta_tensors: List[torch.Tensor], device: str) -> List[torch.Tensor]:
    """Returns a prob tensor per TTA view."""
    model.eval()
    results = []
    with torch.no_grad():
        for t in tta_tensors:
            logits = model(t.to(device))
            p = _apply_temperature(logits, binary=False)  # (1, N)
            results.append(p)
    return results


# ── Main Prediction Pipeline ─────────────────────────────────────────────────

async def run_prediction(image_bytes: bytes) -> PredictionResult:
    device = get_device()
    detection_model = get_detection_model()
    classification_model = get_classification_model()
    threshold = settings.CONFIDENCE_THRESHOLD
    n_mc = 10    # MC Dropout forward passes

    # ResNet101 uses backbone.layer4; EfficientNet-B4 uses backbone.features.7
    def _get_target_layer(model) -> str:
        try:
            _ = model.backbone.layer4
            return "backbone.layer4"
        except AttributeError:
            return "backbone.features.7"

    det_target_layer = _get_target_layer(detection_model)
    cls_target_layer = _get_target_layer(classification_model)

    # ── 1. Preprocess (CLAHE + Skull Stripping) ───────────────────────────────
    pil_img = load_image_from_bytes(image_bytes)
    tensor, np_img = preprocess_for_inference(pil_img)
    tensor = tensor.to(device)
    tta_tensors = [t.to(device) for t in preprocess_tta(pil_img)]

    # ── 2. Detection — TTA pass ───────────────────────────────────────────────
    tta_det_probs = _tta_predict_detection(detection_model, tta_tensors, device)
    tta_det_mean  = float(np.mean(tta_det_probs))
    tta_agreement = float(np.clip(1.0 - np.std(tta_det_probs), 0.0, 1.0))  # higher = more agreement

    # ── 3. Detection — MC Dropout ─────────────────────────────────────────────
    detection_model.eval()
    _enable_mc_dropout(detection_model)
    det_mc_probs = []
    with torch.no_grad():
        for _ in range(n_mc):
            logit = detection_model(tensor).squeeze()
            p = torch.sigmoid(logit / _TEMPERATURE).item()
            det_mc_probs.append(p)

    avg_det = (float(np.mean(det_mc_probs)) + tta_det_mean) / 2.0  # fuse TTA + MC
    det_uncertainty = float(np.std(det_mc_probs))
    tumor_detected = avg_det >= 0.35

    # ── 4. Grad-CAM++ on detection model ─────────────────────────────────────
    detection_model.eval()  # clean state for grad computation
    gcam = GradCAMPlusPlus(detection_model, target_layer=det_target_layer)
    heatmap_b64, _ = gcam.generate(tensor.clone(), np_img, class_idx=None)
    gcam.remove_hooks()

    # ── 5. Score-CAM on detection model ──────────────────────────────────────
    scam = ScoreCAM(detection_model, target_layer=det_target_layer)
    scorecam_b64 = scam.generate(tensor.clone(), np_img, class_idx=None)
    scam.remove_hooks()

    comparison = generate_comparison_strip(np_img, heatmap_b64, scorecam_b64)

    if not tumor_detected:
        # No tumor — binary detection only
        conf = round(1.0 - avg_det, 4)
        det_entropy = round(-avg_det * np.log(avg_det + 1e-8) - (1 - avg_det) * np.log(1 - avg_det + 1e-8), 4)
        is_reliable = conf >= threshold and det_uncertainty < 0.12
        risk = get_risk_report(None)

        # All-class pseudo-probs for UI
        all_probs = {"no_tumor": round(conf, 4)}

        return PredictionResult(
            tumor_detected=False,
            tumor_type=None,
            confidence=conf,
            uncertainty=round(det_uncertainty, 4),
            entropy=det_entropy,
            reliability=(
                "✅ Reliable — Low Uncertainty & High Confidence"
                if is_reliable
                else "⚠️ Uncertain — Consider Repeat Imaging"
            ),
            risk_level=risk.risk_level,
            risk_color=risk.risk_color,
            clinical_note=risk.clinical_note,
            recommendation=risk.recommendation,
            heatmap_image=heatmap_b64,
            scorecam_image=scorecam_b64,
            comparison_strip=comparison,
            all_class_probs=all_probs,
            tta_agreement=round(tta_agreement, 4),
            localization=None,
        )

    # ── 6. Classification — TTA pass ──────────────────────────────────────────
    tta_cls_probs = _tta_predict_classification(classification_model, tta_tensors, device)
    tta_cls_mean = torch.stack(tta_cls_probs).mean(dim=0)  # (1, 3)

    # ── 7. Classification — MC Dropout ───────────────────────────────────────
    classification_model.eval()
    _enable_mc_dropout(classification_model)
    mc_cls_probs = []
    with torch.no_grad():
        for _ in range(n_mc):
            logits = classification_model(tensor)
            p = _apply_temperature(logits, binary=False)
            mc_cls_probs.append(p)

    mc_cls_stack = torch.stack(mc_cls_probs)          # (n_mc, 1, 3)
    avg_cls_probs = mc_cls_stack.mean(dim=0)          # (1, 3)

    # Fuse TTA + MC
    fused_probs = (avg_cls_probs + tta_cls_mean) / 2.0  # (1, 3)
    cls_uncertainties = mc_cls_stack.std(dim=0)          # (1, 3)

    top_idx = fused_probs.argmax(dim=1).item()
    top_conf = fused_probs[0, top_idx].item()
    top_unc  = cls_uncertainties[0, top_idx].item()
    tumor_type = TUMOR_CLASSES[top_idx]

    # Shannon entropy
    cls_entropy = _entropy(fused_probs[0])

    # All-class probs dict for UI
    all_probs = {
        cls: round(fused_probs[0, i].item(), 4)
        for i, cls in enumerate(TUMOR_CLASSES)
    }

    # ── 8. Reliability assessment ─────────────────────────────────────────────
    is_reliable = (
        top_conf >= threshold and
        top_unc < 0.09 and
        cls_entropy < 0.5 and
        tta_agreement > 0.7
    )
    if is_reliable:
        reliability = "✅ Reliable — High Evidence, Low Uncertainty"
    elif top_conf >= threshold and top_unc < 0.15:
        reliability = "🟡 Moderately Reliable — Recommend Clinical Verification"
    else:
        reliability = "🔴 Uncertain — Mandatory Expert Clinical Review"

    # ── 9. Grad-CAM++ for classification ─────────────────────────────────────
    classification_model.eval()
    gcam_cls = GradCAMPlusPlus(classification_model, target_layer=cls_target_layer)
    heatmap_cls_b64, cls_cam = gcam_cls.generate(tensor.clone(), np_img, class_idx=top_idx)
    gcam_cls.remove_hooks()

    # ── 10. Score-CAM for classification ─────────────────────────────────────
    scam_cls = ScoreCAM(classification_model, target_layer=cls_target_layer)
    scorecam_cls_b64 = scam_cls.generate(tensor.clone(), np_img, class_idx=top_idx)
    scam_cls.remove_hooks()

    comparison_cls = generate_comparison_strip(np_img, heatmap_cls_b64, scorecam_cls_b64)

    # ── Localization from CAM ─────────────────────────────────────────────────
    localization = localize_cam(cls_cam, np_img.shape[:2])

    # ── 11. Risk analysis (confidence-weighted) ───────────────────────────────
    risk = get_risk_report(tumor_type, confidence=top_conf, uncertainty=top_unc)

    return PredictionResult(
        tumor_detected=True,
        tumor_type=tumor_type,
        confidence=round(top_conf, 4),
        uncertainty=round(top_unc, 4),
        entropy=cls_entropy,
        reliability=reliability,
        risk_level=risk.risk_level,
        risk_color=risk.risk_color,
        clinical_note=risk.clinical_note,
        recommendation=risk.recommendation,
        heatmap_image=heatmap_cls_b64,
        scorecam_image=scorecam_cls_b64,
        comparison_strip=comparison_cls,
        all_class_probs=all_probs,
        tta_agreement=round(tta_agreement, 4),
        localization=localization,
    )
