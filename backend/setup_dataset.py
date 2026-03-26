"""
setup_dataset.py
─────────────────
Helper script to organise the Kaggle Brain Tumour MRI dataset into the
folder structure expected by train.py.

Kaggle dataset: "Brain Tumor MRI Dataset" by Masoud Nickparvar
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

After downloading and unzipping, run:
  python setup_dataset.py --src /path/to/raw_dataset --dst dataset

Expected raw structure:
  raw/
    Training/
      glioma/
      meningioma/
      notumor/
      pituitary/
    Testing/
      glioma/
      meningioma/
      notumor/
      pituitary/
"""

import os
import shutil
import argparse
import random
from pathlib import Path


def copy_files(src: Path, dst: Path, limit: int | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    files = list(src.glob("*"))
    if limit:
        files = random.sample(files, min(limit, len(files)))
    for f in files:
        shutil.copy2(f, dst / f.name)
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Raw dataset root")
    parser.add_argument("--dst", default="dataset")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    random.seed(42)

    tumor_classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    tumor_class_names = ["glioma", "meningioma", "pituitary"]  # for output naming
    no_tumor_class = "no_tumor"

    print("Setting up DETECTION dataset …")
    for split_name, split_src in [("train", "Training"), ("val", "Testing")]:
        # tumor
        tumor_dst = dst / "detection" / split_name / "tumor"
        count = 0
        for cls in tumor_classes:
            count += copy_files(src / split_src / cls, tumor_dst)
        print(f"  {split_name}/tumor: {count} images")

        # no_tumor
        no_tumor_dst = dst / "detection" / split_name / "no_tumor"
        n = copy_files(src / split_src / no_tumor_class, no_tumor_dst)
        print(f"  {split_name}/no_tumor: {n} images")

    print("\nSetting up CLASSIFICATION dataset …")
    for split_name, split_src in [("train", "Training"), ("val", "Testing")]:
        for cls, cls_name in zip(tumor_classes, tumor_class_names):
            cls_dst = dst / "classification" / split_name / cls_name
            n = copy_files(src / split_src / cls, cls_dst)
            print(f"  {split_name}/{cls_name}: {n} images")

    print(f"\n✅ Dataset organised under '{dst}/'")


if __name__ == "__main__":
    main()
