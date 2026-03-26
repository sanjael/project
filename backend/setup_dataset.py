"""
setup_dataset.py
─────────────────
1. Combines ALL images from Training/ + Testing/ folders
2. Applies random 80/20 train/val split
3. Augments training data automatically:
   - Detection  : no_tumor augmented 8x to balance with tumor
   - Classification: all 3 classes augmented 8x equally

Usage:
  python setup_dataset.py --src /path/to/raw --dst dataset
  python setup_dataset.py --src /path/to/raw --dst dataset --val_split 0.2 --aug_factor 8
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_image(img: Image.Image, idx: int) -> Image.Image:
    """Returns one augmented version of the image based on index."""
    augmentations = [
        lambda i: i.rotate(90),
        lambda i: i.rotate(180),
        lambda i: i.rotate(270),
        lambda i: ImageOps.mirror(i),
        lambda i: ImageOps.flip(i),
        lambda i: ImageEnhance.Brightness(i).enhance(1.3),
        lambda i: ImageEnhance.Contrast(i).enhance(1.3),
        lambda i: ImageEnhance.Brightness(i).enhance(0.8),
    ]
    return augmentations[idx % len(augmentations)](img)


def augment_folder(folder: Path, factor: int):
    """
    Augments all images in a folder by given factor.
    factor=8 means each image produces 8 extra copies -> total 9x original count.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    original_files = [
        f for f in folder.glob("*")
        if f.suffix.lower() in image_exts and not f.stem.startswith("aug_")
    ]

    count = 0
    for img_path in original_files:
        try:
            img = Image.open(img_path).convert("RGB")
            for i in range(factor):
                aug = augment_image(img, i)
                save_path = folder / f"aug_{i}_{img_path.name}"
                aug.save(save_path)
                count += 1
        except Exception as e:
            print(f"  Warning: Could not augment {img_path.name} - {e}")

    return len(original_files), count


# ── Dataset Collection ─────────────────────────────────────────────────────────

def collect_images(src: Path, class_folders: list) -> list:
    """Collect all images from Training/ and Testing/ for given class folders."""
    images = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for split in ["Training", "Testing"]:
        for folder in class_folders:
            folder_path = src / split / folder
            if folder_path.exists():
                for f in folder_path.iterdir():
                    if f.suffix.lower() in image_exts:
                        images.append(f)
    return images


def split_and_copy(images: list, train_dst: Path, val_dst: Path, val_split: float):
    """Randomly split and copy images to train/val folders."""
    random.shuffle(images)
    split_idx  = int(len(images) * (1 - val_split))
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    for f in train_imgs:
        dst = train_dst / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    for f in val_imgs:
        dst = val_dst / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    return len(train_imgs), len(val_imgs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",        required=True,       help="Raw dataset root (contains Training/ and Testing/)")
    parser.add_argument("--dst",        default="dataset",   help="Output dataset directory")
    parser.add_argument("--val_split",  type=float, default=0.2, help="Val split ratio (default: 0.2)")
    parser.add_argument("--aug_factor", type=int,   default=8,   help="Augmentation multiplier (default: 8)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    random.seed(42)

    # Clear existing dataset
    if dst.exists():
        shutil.rmtree(dst)
        print(f"Cleared existing dataset at '{dst}/'")

    val_pct   = int(args.val_split * 100)
    train_pct = 100 - val_pct
    print(f"\nSplit     : {train_pct}% train / {val_pct}% val")
    print(f"Aug factor: {args.aug_factor}x (each image -> {args.aug_factor + 1} total)")
    print(f"Seed      : 42 (reproducible)\n")

    tumor_folders    = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    tumor_names      = ["glioma",       "meningioma",       "pituitary"]
    no_tumor_folder  = ["no_tumor"]

    # ── DETECTION ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("DETECTION DATASET")
    print("=" * 55)

    # tumor
    tumor_imgs = collect_images(src, tumor_folders)
    tr, vl = split_and_copy(
        tumor_imgs,
        dst / "detection" / "train" / "tumor",
        dst / "detection" / "val"   / "tumor",
        args.val_split,
    )
    print(f"  tumor    — collected: {len(tumor_imgs)} | train: {tr} | val: {vl}")

    # no_tumor
    no_tumor_imgs = collect_images(src, no_tumor_folder)
    tr, vl = split_and_copy(
        no_tumor_imgs,
        dst / "detection" / "train" / "no_tumor",
        dst / "detection" / "val"   / "no_tumor",
        args.val_split,
    )
    print(f"  no_tumor — collected: {len(no_tumor_imgs)} | train: {tr} | val: {vl}")

    # Augment detection train
    print(f"\n  Augmenting detection/train/tumor ({args.aug_factor}x)...")
    orig, added = augment_folder(dst / "detection" / "train" / "tumor", args.aug_factor)
    total_tumor = orig + added
    print(f"    tumor    : {orig} original + {added} augmented = {total_tumor} total")

    print(f"  Augmenting detection/train/no_tumor ({args.aug_factor}x)...")
    orig, added = augment_folder(dst / "detection" / "train" / "no_tumor", args.aug_factor)
    total_no_tumor = orig + added
    print(f"    no_tumor : {orig} original + {added} augmented = {total_no_tumor} total")

    # ── CLASSIFICATION ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("CLASSIFICATION DATASET")
    print("=" * 55)

    for folder, name in zip(tumor_folders, tumor_names):
        imgs = collect_images(src, [folder])
        tr, vl = split_and_copy(
            imgs,
            dst / "classification" / "train" / name,
            dst / "classification" / "val"   / name,
            args.val_split,
        )
        print(f"  {name:<12} — collected: {len(imgs)} | train: {tr} | val: {vl}")

    print(f"\n  Augmenting classification train ({args.aug_factor}x)...")
    for name in tumor_names:
        orig, added = augment_folder(
            dst / "classification" / "train" / name, args.aug_factor
        )
        total = orig + added
        print(f"    {name:<12}: {orig} original + {added} augmented = {total} total")

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("FINAL DATASET SUMMARY")
    print("=" * 55)

    grand_train = 0
    grand_val   = 0

    for task in ["detection", "classification"]:
        print(f"\n  [{task.upper()}]")
        for split in ["train", "val"]:
            split_path = dst / task / split
            if split_path.exists():
                for cls in sorted(os.listdir(split_path)):
                    cls_path = split_path / cls
                    count = len(list(cls_path.glob("*")))
                    print(f"    {split}/{cls:<15}: {count:>6} images")
                    if split == "train":
                        grand_train += count
                    else:
                        grand_val += count

    print(f"\n  Total Train : {grand_train}")
    print(f"  Total Val   : {grand_val}")
    print(f"  Grand Total : {grand_train + grand_val}")
    print(f"\nDataset ready at '{dst}/'")


if __name__ == "__main__":
    main()
