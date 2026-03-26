"""
models/resnet_models.py
────────────────────────
ResNet101-based models for:
  1. Tumor Detection  (binary: tumor / no_tumor)
  2. Tumor Classification  (3-class: glioma / meningioma / pituitary)

Both models fine-tune a pretrained ResNet101 backbone with a custom head.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights


# ── Detection Model ───────────────────────────────────────────────────────────

class TumorDetectionModel(nn.Module):
    """
    Binary classifier:  0 = no_tumor,  1 = tumor
    Optimised for HIGH RECALL to minimise false negatives.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet101(weights=weights)

        # Freeze early layers (stem + layer1) — fine-tune deeper layers
        for name, param in backbone.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1")):
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),          # raw logit
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)          # shape: (B, 1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability of TUMOR class."""
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)  # (B,)


# ── Classification Model ──────────────────────────────────────────────────────

TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]


class TumorClassificationModel(nn.Module):
    """
    3-class classifier for tumour type.
    Input: MRI confirmed to contain a tumour.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 3):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet101(weights=weights)

        for name, param in backbone.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),   # raw logits
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)            # shape: (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities over 3 classes."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)  # (B, 3)


# ── Utility: build models ─────────────────────────────────────────────────────

def build_detection_model(weights_path: str | None = None, device: str = "cpu") -> TumorDetectionModel:
    model = TumorDetectionModel(pretrained=(weights_path is None))
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_classification_model(weights_path: str | None = None, device: str = "cpu") -> TumorClassificationModel:
    model = TumorClassificationModel(pretrained=(weights_path is None))
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
