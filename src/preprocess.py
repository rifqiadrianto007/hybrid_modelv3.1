from __future__ import annotations

import argparse
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
RESAMPLE_METHOD = Image.Resampling.LANCZOS
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def simple_crop_center(img, crop_frac : float = 0.9) :
    w, h = img.size
    crop = int(min(w, h) * crop_frac)
    left = (w - crop) // 2
    top = (h - crop) // 2

    return img.crop((left, top, left + crop, top + crop))

def random_crop(img, crop_frac_min : float = 0.8, crop_frac_max : float = 1.0) :
    w, h = img.size
    short = min(w, h)
    frac = random.uniform(crop_frac_min, crop_frac_max)
    crop = int(short * frac)

    if w == short :
        left = (0 if w == crop else random.randint(0, w - crop))
        top = random.randint(0, h - crop)
    else :
        left = random.randint(0, w - crop)
        top = (0 if h == crop else random.randint(0, h - crop))

    return img.crop((left, top, left + crop, top + crop))

def load_and_resize(path : Path, split : str = "train") :
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if split == "train" :
        img = random_crop(img)
    else :
        img = simple_crop_center(img)

    return img.resize(IMG_SIZE, RESAMPLE_METHOD)

def save(img, path) :
    path.parent.mkdir(parents = True, exist_ok = True)

    img.save(path, quality = 95)

def process_split(split, input_dir, output_dir) :
    if not input_dir.exists() :
        return
    if output_dir.exists() :
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents = True, exist_ok = True)

    for cls in CLASS_NAMES :
        src = input_dir / cls
        dst = output_dir / cls

        if not src.exists() :
            continue

        for file in src.rglob("*") :

            if file.suffix.lower() not in VALID_EXT :
                continue
            img = load_and_resize(file, split = split)
            save(img, dst / f"{file.stem}.jpg")

def count_images(split_dir) :
    if not split_dir.exists() :
        return 0

    return sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in VALID_EXT)

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type = str, default = None)
    args = parser.parse_args()
    base_dir = (Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parents[1])
    split_root = (base_dir / "data" / "splits")
    output_root = (base_dir / "data" / "processed")

    random.seed(SEED)
    np.random.seed(SEED)

    for split in ["train", "validation", "test"] :
        print(f"Processing {split}...")
        process_split(
            split = split,
            input_dir = split_root / split,
            output_dir = output_root / split
        )

    train_count = count_images(output_root / "train")
    val_count = count_images(output_root / "validation")
    test_count = count_images(output_root / "test")

    print("\nHASIL PREPROCESSING")
    print(f"Train      : {train_count}")
    print(f"Validation : {val_count}")
    print(f"Test       : {test_count}")
    print(
        f"Total      : "
        f"{train_count + val_count + test_count}"
    )

if __name__ == "__main__":
    main()