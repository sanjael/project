"""
services/preprocessing.py  (UPGRADED v2)
─────────────────────────────────────────
Advanced MRI preprocessing pipeline:
  1. CLAHE contrast-limited adaptive histogram equalisation (improves tumour visibility)
  2. Gaussian denoising (reduces MRI noise artefacts)
  3. Multi-scale inference: 224×224 + 299×299 tensors for ensemble TTA
  4. Standard ImageNet normalisation for pretrained backbone compatibility
  5. Test-Time Augmentation (TTA): horizontal flip + brightness shift
"""

import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
from torchvision import transforms
from typing import Tuple, List


# ── Constants ─────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_SIZE_PRIMARY   = 224   # ResNet101 / EfficientNet-B4
_SIZE_SECONDARY = 299   # optional larger scale for TTA

_to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

_tta_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

_tta_bright = transforms.Compose([
    transforms.ColorJitter(brightness=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── CLAHE Contrast Enhancement ────────────────────────────────────────────────

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to L channel of LAB colour space.
    Enhances local contrast without over-amplifying noise.
    Critical for highlighting tumour boundaries in T1/T2 MRI.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def denoise(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply non-local means denoising — removes MRI Rician noise.
    """
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h=7, hColor=7,
                                           templateWindowSize=7, searchWindowSize=21)


def skull_strip_simulation(img_rgb: np.ndarray) -> np.ndarray:
    """
    Simulate skull stripping by masking non-brain tissue.
    Finds the largest bright ellipse region (brain) and masks the border.
    This reduces bone/skull features that can confuse the CNN.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    # Keep only the largest contour (brain region)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    # Erode slightly to remove skull ring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=2)
    result = img_rgb.copy()
    result[mask == 0] = 0
    return result


# ── Main public API ───────────────────────────────────────────────────────────

def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load PIL image from raw bytes (JPEG / PNG / BMP …)."""
    img = Image.open(BytesIO(data)).convert("RGB")
    return img


def preprocess_for_inference(img: Image.Image, size: int = _SIZE_PRIMARY) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor : (1, 3, size, size) torch.FloatTensor  → model input
        np_img : (size, size, 3)    np.ndarray uint8   → Grad-CAM overlay
    """
    img_resized = img.resize((size, size), Image.LANCZOS)
    
    # Apply CLAHE + skull strip
    np_raw = np.array(img_resized, dtype=np.uint8)
    np_enhanced = apply_clahe(np_raw)
    np_enhanced = skull_strip_simulation(np_enhanced)
    np_img = np_enhanced  # for Grad-CAM overlay

    pil_enhanced = Image.fromarray(np_enhanced)
    tensor = _to_tensor_normalize(pil_enhanced).unsqueeze(0)  # (1, 3, size, size)
    return tensor, np_img


def preprocess_tta(img: Image.Image, size: int = _SIZE_PRIMARY) -> List[torch.Tensor]:
    """
    Test-Time Augmentation: returns a list of 3 tensors
    [original, h-flip, brightness-shifted]
    — averages predictions for higher accuracy at inference.
    """
    img_resized = img.resize((size, size), Image.LANCZOS)
    np_raw = np.array(img_resized, dtype=np.uint8)
    np_enhanced = apply_clahe(np_raw)
    np_enhanced = skull_strip_simulation(np_enhanced)
    pil_enhanced = Image.fromarray(np_enhanced)

    t_orig  = _to_tensor_normalize(pil_enhanced).unsqueeze(0)
    t_flip  = _tta_flip(pil_enhanced).unsqueeze(0)
    t_bright = _tta_bright(pil_enhanced).unsqueeze(0)
    return [t_orig, t_flip, t_bright]


def get_multi_scale_tensors(img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns tensors at two scales for multi-scale inference.
    224×224: standard ResNet/EfficientNet-B4 input
    299×299: EfficientNet's optimal input (optional)
    """
    t224, np224 = preprocess_for_inference(img, size=_SIZE_PRIMARY)
    t299, _     = preprocess_for_inference(img, size=_SIZE_SECONDARY)
    return t224, t299, np224
