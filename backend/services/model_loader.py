"""
services/model_loader.py  (UPGRADED v2)
─────────────────────────────────────────
Extended loader that supports:
  - ResNet101 models (original v1)
  - EfficientNet-B4 models (new v2, preferred if weights exist)
  - Automatic fallback: if EfficientNet weights not found, use ResNet101
  - Thread-safe singleton pattern
"""

import os
import threading
import torch
from typing import Optional

from config import get_settings
from models.resnet_models import (
    TumorDetectionModel,
    TumorClassificationModel,
    build_detection_model,
    build_classification_model,
)
from models.advanced_models import (
    EfficientDetectionModel,
    EfficientClassificationModel,
    build_efficient_detection,
    build_efficient_classification,
)

settings = get_settings()
_lock = threading.Lock()

_detection_model = None
_classification_model = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _load_models():
    global _detection_model, _classification_model

    # ── Paths ──────────────────────────────────────────────────────────────────
    # EfficientNet-B4 weights (preferred — higher accuracy)
    eff_det_path = "models/detection_efficientnet_b4.pth"
    eff_cls_path = "models/classification_efficientnet_b4.pth"

    # ResNet101 weights (fallback)
    res_det_path = settings.MODEL_DETECTION_PATH       # models/detection_resnet101.pth
    res_cls_path = settings.MODEL_CLASSIFICATION_PATH  # models/classification_resnet101.pth

    # ── Detection Model ────────────────────────────────────────────────────────
    if os.path.exists(eff_det_path):
        print(f"[ModelLoader] Loading EfficientNet-B4 detection model from {eff_det_path}")
        _detection_model = build_efficient_detection(eff_det_path, _device)
    elif os.path.exists(res_det_path):
        print(f"[ModelLoader] Loading ResNet101 detection model from {res_det_path}")
        _detection_model = build_detection_model(res_det_path, _device)
    else:
        print("[ModelLoader] WARNING: No trained detection weights found — using ImageNet pretrained EfficientNet-B4 (demo mode).")
        _detection_model = build_efficient_detection(None, _device)

    # ── Classification Model ───────────────────────────────────────────────────
    if os.path.exists(eff_cls_path):
        print(f"[ModelLoader] Loading EfficientNet-B4 classification model from {eff_cls_path}")
        _classification_model = build_efficient_classification(eff_cls_path, _device)
    elif os.path.exists(res_cls_path):
        print(f"[ModelLoader] Loading ResNet101 classification model from {res_cls_path}")
        _classification_model = build_classification_model(res_cls_path, _device)
    else:
        print("[ModelLoader] WARNING: No trained classification weights found — using ImageNet pretrained EfficientNet-B4 (demo mode).")
        _classification_model = build_efficient_classification(None, _device)

    print(f"[ModelLoader] Models ready on device: {_device}")


def get_detection_model():
    global _detection_model
    if _detection_model is None:
        with _lock:
            if _detection_model is None:
                _load_models()
    return _detection_model


def get_classification_model():
    global _classification_model
    if _classification_model is None:
        with _lock:
            if _classification_model is None:
                _load_models()
    return _classification_model


def get_device() -> str:
    return _device


def reload_models():
    """Force model reload — useful after training new weights."""
    global _detection_model, _classification_model
    with _lock:
        _detection_model = None
        _classification_model = None
        _load_models()
    print("[ModelLoader] Models reloaded.")
