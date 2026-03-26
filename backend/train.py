"""
train.py — ADVANCED training pipeline  (v2)
────────────────────────────────────────────
Improvements over v1:
  ✅ Focal Loss (handles class imbalance better than BCE)
  ✅ Label Smoothing Cross-Entropy (reduces overconfidence)
  ✅ MixUp augmentation (improves generalisation)
  ✅ CutMix augmentation (better feature localisation)
  ✅ OneCycleLR scheduler (cosine warm-up + annealing)
  ✅ Class-weighted sampling (handles imbalanced datasets)
  ✅ Gradient clipping (prevents exploding gradients)
  ✅ EarlyStopping (prevents overfitting)
  ✅ Best model saved by F1 (not just accuracy — fairer for imbalanced)
  ✅ Support for EfficientNet-B4 model (higher accuracy)
  ✅ Comprehensive metrics: AUC-ROC, per-class F1, confusion matrix

Dataset structure expected:
  dataset/
    detection/
      train/tumor/, no_tumor/
      val/tumor/, no_tumor/
    classification/
      train/glioma/, meningioma/, pituitary/
      val/glioma/, meningioma/, pituitary/

Run examples:
  python train.py --task detection --model efficientnet --epochs 40
  python train.py --task classification --model efficientnet --epochs 40 --mixup
  python train.py --task detection --model resnet101 --epochs 30 --focal
"""

import os
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score,
)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.resnet_models import TumorDetectionModel, TumorClassificationModel, TUMOR_CLASSES
from models.advanced_models import EfficientDetectionModel, EfficientClassificationModel

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 224
BATCH_SIZE  = 16   # smaller batch → better generalisation for EfficientNet
NUM_WORKERS = 0    # set to 4 on Linux/Mac; 0 on Windows to avoid multiprocessing issues
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]


# ── Advanced Augmentation Transforms ──────────────────────────────────────────

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # simulates MRI artefacts
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard misclassified samples.
    Excellent for imbalanced medical datasets.
    gamma=2 is standard; higher gamma = more focus on hard examples.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, binary: bool = False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.binary = binary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.binary:
            bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
            pt  = torch.exp(-bce)
            focal_weight = self.alpha * (1 - pt) ** self.gamma
            return (focal_weight * bce).mean()
        else:
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()


# ── Label Smoothing Cross-Entropy ─────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    """Reduces overconfidence by assigning small probability to wrong classes."""
    def __init__(self, smoothing: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, smooth_val)
            smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
        log_prob = F.log_softmax(logits, dim=1)
        return -(smooth_targets * log_prob).sum(dim=1).mean()


# ── MixUp Augmentation ────────────────────────────────────────────────────────

def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """
    MixUp: blend pairs of training examples.
    Returns mixed images + both sets of labels + lambda.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bs = images.size(0)
    idx = torch.randperm(bs, device=images.device)
    mixed_x = lam * images + (1 - lam) * images[idx]
    return mixed_x, labels, labels[idx], lam


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """
    CutMix: paste random patch from one image to another.
    More beneficial than MixUp for localised features like tumours.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bs, _, h, w = images.shape
    idx = torch.randperm(bs, device=images.device)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)

    mixed_x = images.clone()
    mixed_x[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
    return mixed_x, labels, labels[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, binary=False):
    if binary:
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Class-Weighted Sampler ────────────────────────────────────────────────────

def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Data Loaders ──────────────────────────────────────────────────────────────

def get_loaders(data_dir: str, weighted: bool = True):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_train_transform())
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=get_val_transform())

    sampler = make_weighted_sampler(train_ds) if weighted else None
    shuffle = (sampler is None)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    print(f"  Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"  Classes: {train_ds.classes}")
    return train_loader, val_loader, train_ds.classes


# ── Training loop ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, score: float):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_one_epoch(model, loader, criterion, optimizer, binary: bool,
                    use_mixup: bool = False, use_cutmix: bool = False,
                    clip_grad: float = 1.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        aug_choice = random.random()
        if use_cutmix and aug_choice < 0.4:
            imgs, y_a, y_b, lam = cutmix_batch(imgs, labels, alpha=0.5)
            mixed = True
        elif use_mixup and aug_choice < 0.7:
            imgs, y_a, y_b, lam = mixup_batch(imgs, labels, alpha=0.4)
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad()

        if binary:
            outputs = model(imgs).squeeze(1)
            if mixed:
                loss = mixup_criterion(criterion, outputs, y_a.float(), y_b.float(), lam, binary=True)
            else:
                loss = criterion(outputs, labels.float())
            preds = (torch.sigmoid(outputs) >= 0.5).long()
        else:
            outputs = model(imgs)
            if mixed:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, binary: bool):
    model.eval()
    running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if binary:
                outputs = model(imgs).squeeze(1)
                loss = criterion(outputs, labels.float())
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_probs.extend(probs)

            running_loss += loss.item() * imgs.size(0)
            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else preds)
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = running_loss / n
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    # AUC-ROC
    try:
        if binary:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    return avg_loss, acc, prec, rec, f1, auc, cm, all_preds, all_labels


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, key_tr, key_vl, title in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_acc",  "val_acc",  "Accuracy"),
        (axes[2], "val_f1",     "val_auc",  "F1 & AUC (val)"),
    ]:
        ax.plot(epochs, history[key_tr], label=key_tr.replace("_", " ").title())
        ax.plot(epochs, history[key_vl], label=key_vl.replace("_", " ").title())
        ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Training curves → {save_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced Brain Tumour Model Trainer v2")
    parser.add_argument("--task",    choices=["detection", "classification"], required=True)
    parser.add_argument("--model",   choices=["resnet101", "efficientnet"],   default="efficientnet")
    parser.add_argument("--data",    default="dataset",  help="Root dataset directory")
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--out",     default="models",   help="Output directory for weights")
    parser.add_argument("--mixup",   action="store_true", help="Enable MixUp augmentation")
    parser.add_argument("--cutmix",  action="store_true", help="Enable CutMix augmentation")
    parser.add_argument("--focal",   action="store_true", help="Use Focal Loss")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(args.data, args.task)
    binary = (args.task == "detection")

    print(f"\n{'='*65}")
    print(f"  Task     : {args.task.upper()}")
    print(f"  Model    : {args.model.upper()}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  MixUp    : {args.mixup}  |  CutMix: {args.cutmix}  |  Focal: {args.focal}")
    print(f"{'='*65}\n")

    train_loader, val_loader, classes = get_loaders(data_dir, weighted=True)

    # ── Build model ────────────────────────────────────────────────────────────
    if args.model == "efficientnet":
        if binary:
            model = EfficientDetectionModel(pretrained=True).to(DEVICE)
            out_path = os.path.join(args.out, "detection_efficientnet_b4.pth")
        else:
            model = EfficientClassificationModel(pretrained=True, num_classes=len(classes)).to(DEVICE)
            out_path = os.path.join(args.out, "classification_efficientnet_b4.pth")
    else:
        if binary:
            model = TumorDetectionModel(pretrained=True).to(DEVICE)
            out_path = os.path.join(args.out, "detection_resnet101.pth")
        else:
            model = TumorClassificationModel(pretrained=True, num_classes=len(classes)).to(DEVICE)
            out_path = os.path.join(args.out, "classification_resnet101.pth")

    # ── Loss function ──────────────────────────────────────────────────────────
    if args.focal:
        criterion = FocalLoss(gamma=2.0, alpha=0.75, binary=binary)
    elif binary:
        # Strong positive weight for high recall (critical for medical detection)
        pos_w = torch.tensor([3.0]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        criterion = LabelSmoothingCE(smoothing=0.1, num_classes=len(classes))

    # ── Optimizer: differential learning rates ─────────────────────────────────
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # lower LR for pretrained backbone
        {"params": head_params,     "lr": args.lr},          # higher LR for new head
    ], weight_decay=1e-4)

    # ── Scheduler: OneCycleLR (cosine warm-up + annealing) ────────────────────
    steps_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        steps_per_epoch=steps_epoch,
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_f1": [], "val_auc": [],
    }
    best_val_f1 = 0.0
    early_stop = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, binary,
            use_mixup=args.mixup, use_cutmix=args.cutmix, clip_grad=1.0,
        )
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, vl_auc, cm, preds, labels = evaluate(
            model, val_loader, criterion, binary
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)
        history["val_auc"].append(vl_auc)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f} | "
            f"VlLoss={vl_loss:.4f} VlAcc={vl_acc:.4f} "
            f"F1={vl_f1:.4f} AUC={vl_auc:.4f}"
        )

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), out_path)
            print(f"  ✅ Best model saved (F1={best_val_f1:.4f}) → {out_path}")

        early_stop(vl_f1)
        if early_stop.stop:
            print(f"  ⏹ Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n🏁 Training complete. Best Val F1: {best_val_f1:.4f}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(out_path, map_location=DEVICE, weights_only=True))
    _, fin_acc, fin_prec, fin_rec, fin_f1, fin_auc, fin_cm, fin_preds, fin_labels = evaluate(
        model, val_loader, criterion, binary
    )

    print("\n📋 Final Classification Report:")
    print(classification_report(fin_labels, fin_preds, target_names=classes))
    print(f"  AUC-ROC: {fin_auc:.4f}")

    prefix = os.path.join(args.out, f"{args.task}_{args.model}")
    plot_history(history, prefix + "_training_curves.png")
    plot_confusion_matrix(fin_cm, classes, prefix + "_confusion_matrix.png")

    metrics = {
        "task": args.task,
        "model": args.model,
        "best_val_f1": best_val_f1,
        "final_accuracy": fin_acc,
        "final_precision": fin_prec,
        "final_recall": fin_rec,
        "final_f1": fin_f1,
        "final_auc_roc": fin_auc,
        "classes": classes,
        "augmentation": {"mixup": args.mixup, "cutmix": args.cutmix, "focal_loss": args.focal},
    }
    with open(prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄 Metrics → {prefix}_metrics.json")


if __name__ == "__main__":
    main()
