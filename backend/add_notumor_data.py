"""
add_notumor_data.py
────────────────────
1. Copies new no_tumor images from archive3_extracted into Training/no_tumor
2. Re-runs setup_dataset.py to rebuild dataset with proper 80/20 train/val split
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path("e:/D folder/Downloads/brain_tumor_project/brain_tumor_project")
NEW_SRC      = Path("e:/D folder/Downloads/archive3_extracted/no")
TRAIN_DST    = ROOT / "Training" / "no_tumor"
TEST_DST     = ROOT / "Testing"  / "no_tumor"
BACKEND      = ROOT / "backend"
DATASET_DST  = BACKEND / "dataset"

# ── Step 1: Copy new images into Training/no_tumor ────────────────────────────
print("Step 1 — Copying new no_tumor images...")
new_files = list(NEW_SRC.glob("*"))
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
new_images = [f for f in new_files if f.suffix.lower() in image_exts]

copied = 0
skipped = 0
for img in new_images:
    dst = TRAIN_DST / f"new_{img.name}"
    if not dst.exists():
        shutil.copy2(img, dst)
        copied += 1
    else:
        skipped += 1

print(f"  [OK] Copied: {copied} | Skipped (already exist): {skipped}")

# ── Step 2: Count all available no_tumor images ───────────────────────────────
all_train = list(TRAIN_DST.glob("*"))
all_test  = list(TEST_DST.glob("*"))
print(f"\nStep 2 — Current counts:")
print(f"  Training/no_tumor : {len(all_train)} images")
print(f"  Testing/no_tumor  : {len(all_test)} images")

# ── Step 3: Rebuild dataset with 80/20 split ─────────────────────────────────
print("\nStep 3 — Rebuilding dataset with 80/20 train/val split...")

# Combine all no_tumor images
all_no_tumor = list(TRAIN_DST.glob("*")) + list(TEST_DST.glob("*"))
all_no_tumor = [f for f in all_no_tumor if f.suffix.lower() in image_exts]
random.shuffle(all_no_tumor)

split_idx  = int(len(all_no_tumor) * 0.8)
train_imgs = all_no_tumor[:split_idx]
val_imgs   = all_no_tumor[split_idx:]

# Detection dataset paths
det_train_no  = DATASET_DST / "detection" / "train" / "no_tumor"
det_val_no    = DATASET_DST / "detection" / "val"   / "no_tumor"

# Clear existing
for folder in [det_train_no, det_val_no]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

# Copy train split
for img in train_imgs:
    shutil.copy2(img, det_train_no / img.name)

# Copy val split
for img in val_imgs:
    shutil.copy2(img, det_val_no / img.name)

print(f"  [OK] detection/train/no_tumor : {len(train_imgs)} images")
print(f"  [OK] detection/val/no_tumor   : {len(val_imgs)} images")

# ── Step 4: Show final dataset summary ───────────────────────────────────────
print("\n" + "="*50)
print("FINAL DATASET SUMMARY")
print("="*50)

splits = {
    "detection/train/tumor"    : DATASET_DST / "detection" / "train" / "tumor",
    "detection/train/no_tumor" : DATASET_DST / "detection" / "train" / "no_tumor",
    "detection/val/tumor"      : DATASET_DST / "detection" / "val"   / "tumor",
    "detection/val/no_tumor"   : DATASET_DST / "detection" / "val"   / "no_tumor",
}

total_train = 0
total_val   = 0
for name, path in splits.items():
    count = len(list(path.glob("*"))) if path.exists() else 0
    print(f"  {name:<35} : {count} images")
    if "train" in name:
        total_train += count
    else:
        total_val += count

print(f"\n  Total Train : {total_train}")
print(f"  Total Val   : {total_val}")
print(f"  Grand Total : {total_train + total_val}")
print("\n[DONE] Dataset ready! Now upload to Colab and run training.")
