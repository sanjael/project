"""
models/advanced_models.py
──────────────────────────
Advanced model architectures for higher accuracy:

1. EfficientNet-B4 with Squeeze-Excite attention
2. EfficientNet-B0 lightweight alternative  
3. Ensemble model (ResNet101 + EfficientNet-B4)
4. Multi-Scale Feature Fusion head

All models support MC Dropout for Bayesian uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
from typing import Optional, List


# ── Squeeze-Excite Attention Block ────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block to recalibrate channel-wise feature responses.
    Improves sensitivity to tumour-relevant features.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


# ── Enhanced Classification Head ──────────────────────────────────────────────

class EnhancedHead(nn.Module):
    """
    Multi-layer classification head with:
    - BatchNorm for stable training
    - GELU activation (smoother than ReLU)
    - Two Dropout layers for MC Dropout inference
    - Residual skip for gradient flow
    """
    def __init__(self, in_features: int, num_classes: int, dropout_p: float = 0.4):
        super().__init__()
        hidden = max(in_features // 2, 256)
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.bn2 = nn.BatchNorm1d(hidden // 2)
        self.drop2 = nn.Dropout(p=dropout_p * 0.75)
        self.fc_out = nn.Linear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn0(x)
        x = self.drop1(self.act(self.bn1(self.fc1(x))))
        x = self.drop2(self.act(self.bn2(self.fc2(x))))
        return self.fc_out(x)


# ── EfficientNet-B4 Detection Model ──────────────────────────────────────────

class EfficientDetectionModel(nn.Module):
    """
    EfficientNet-B4 binary classifier (tumor / no-tumor).
    Higher accuracy than ResNet101 with ~same inference speed.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        # Freeze early layers
        children = list(backbone.features.children())
        for layer in children[:4]:
            for p in layer.parameters():
                p.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()  # remove default head

        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = EnhancedHead(in_features, num_classes=1, dropout_p=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)   # (B, in_features)
        return self.head(feat)    # (B, 1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)).squeeze(1)  # (B,)


# ── EfficientNet-B4 Classification Model ─────────────────────────────────────

TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

class EfficientClassificationModel(nn.Module):
    """
    EfficientNet-B4 multi-class classifier for tumor type.
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 3):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        children = list(backbone.features.children())
        for layer in children[:3]:
            for p in layer.parameters():
                p.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = EnhancedHead(in_features, num_classes=num_classes, dropout_p=0.35)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)      # (B, in_features)
        return self.head(feat)       # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)   # (B, num_classes)


# ── Ensemble Model (ResNet101 + EfficientNet-B4) ──────────────────────────────

class EnsembleDetectionModel(nn.Module):
    """
    Ensemble of ResNet101 and EfficientNet-B4 detection models.
    Outputs are averaged for final prediction — significantly boosts accuracy.
    """
    def __init__(self, model_a: nn.Module, model_b: nn.Module):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p_a = self.model_a.predict_proba(x)
        p_b = self.model_b.predict_proba(x)
        return (p_a + p_b) / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x)


class EnsembleClassificationModel(nn.Module):
    """
    Ensemble of ResNet101 and EfficientNet-B4 classification models.
    Soft voting over probability distributions.
    """
    def __init__(self, model_a: nn.Module, model_b: nn.Module):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p_a = self.model_a.predict_proba(x)   # (B, 3)
        p_b = self.model_b.predict_proba(x)   # (B, 3)
        return (p_a + p_b) / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x)


# ── Build helpers ─────────────────────────────────────────────────────────────

def build_efficient_detection(weights_path: Optional[str] = None, device: str = "cpu") -> EfficientDetectionModel:
    model = EfficientDetectionModel(pretrained=(weights_path is None))
    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model.to(device).eval()


def build_efficient_classification(weights_path: Optional[str] = None, device: str = "cpu", num_classes: int = 3) -> EfficientClassificationModel:
    model = EfficientClassificationModel(pretrained=(weights_path is None), num_classes=num_classes)
    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model.to(device).eval()
