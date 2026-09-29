from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
RESAMPLE_METHOD = Image.Resampling.LANCZOS
SEED = 42
SPLITS = ("train", "validation", "test")

def center_crop(image: Image.Image, crop_frac: float) -> Image.Image:
    width, height = image.size
    crop_size = max(1, int(min(width, height) * crop_frac))
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))

def load_and_resize(path: Path, crop_frac: float) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = center_crop(image, crop_frac)
        return image.resize(IMG_SIZE, RESAMPLE_METHOD)

def process_manifest(manifest_path: Path, base_dir: Path, output_dir: Path, crop_frac: float) -> int:
    manifest = pd.read_csv(manifest_path)
    required_columns = {"image_id", "filepath", "label"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Kolom manifest {manifest_path} tidak lengkap: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest.itertuples(index=False):
        source_path = base_dir / Path(row.filepath)
        output_path = output_dir / str(row.label) / f"{row.image_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = load_and_resize(source_path, crop_frac)
        image.save(output_path, format="PNG")
    return len(manifest)

def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digests(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): file_digest(path)
        for path in sorted(directory.rglob("*.png"))
    }


def verify_processed(manifest_path: Path, output_dir: Path) -> int:
    manifest = pd.read_csv(manifest_path)
    expected_paths = {
        f"{row.label}/{row.image_id}.png"
        for row in manifest.itertuples(index=False)
    }
    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*.png")
    }
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise AssertionError(f"File hasil tidak cocok. Missing={missing[:3]}, extra={extra[:3]}")
    for relative in sorted(actual_paths):
        path = output_dir / relative
        with Image.open(path) as image:
            if image.size != IMG_SIZE or image.mode != "RGB":
                raise AssertionError(f"Format hasil tidak valid: {path} ({image.size}, {image.mode})")
    return len(actual_paths)


def run_determinism_check(manifests: dict[str, Path], base_dir: Path, crop_frac: float) -> None:
    with tempfile.TemporaryDirectory(prefix="preprocess_determinism_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        first_root = temporary_root / "first"
        second_root = temporary_root / "second"
        for split, manifest_path in manifests.items():
            process_manifest(manifest_path, base_dir, first_root / split, crop_frac)
            process_manifest(manifest_path, base_dir, second_root / split, crop_frac)
        if directory_digests(first_root) != directory_digests(second_root):
            raise AssertionError("Uji determinisme gagal: dua hasil preprocessing berbeda.")


def save_montage(train_dir: Path, output_path: Path) -> None:
    image_paths = sorted(train_dir.rglob("*.png"))
    if not image_paths:
        raise ValueError("Tidak ada hasil train untuk dibuat montase.")
    sample_count = min(12, len(image_paths))
    generator = np.random.default_rng(SEED)
    selected_indices = generator.choice(len(image_paths), size=sample_count, replace=False)
    selected_paths = [image_paths[index] for index in selected_indices]
    columns = 4
    rows = (sample_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, path in zip(axes, selected_paths):
        with Image.open(path) as image:
            axis.imshow(image)
        axis.set_title(path.parent.name, fontsize=9)
        axis.axis("off")
    for axis in axes[sample_count:]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing deterministic dari manifest CSV.")
    parser.add_argument("--tag", choices=["pilot", "full"], required=True)
    parser.add_argument("--crop-frac", type=float, default=1.0)
    parser.add_argument("--base-dir", type=str, default=None)
    args = parser.parse_args()
    if not 0 < args.crop_frac <= 1:
        raise ValueError("--crop-frac harus lebih besar dari 0 dan paling besar 1.")

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    split_root = base_dir / "data" / "splits" / args.tag
    output_root = base_dir / "data" / "processed"
    report_root = base_dir / "reports"
    manifests = {split: split_root / f"{split}.csv" for split in SPLITS}
    missing_manifests = [str(path) for path in manifests.values() if not path.exists()]
    if missing_manifests:
        raise FileNotFoundError(f"Manifest tidak ditemukan: {missing_manifests}")

    run_determinism_check(manifests, base_dir, args.crop_frac)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split, manifest_path in manifests.items():
        print(f"Processing {split}...")
        split_output = output_root / split
        counts[split] = process_manifest(manifest_path, base_dir, split_output, args.crop_frac)
        verify_processed(manifest_path, split_output)
        class_counts[split] = pd.read_csv(manifest_path)["label"].value_counts().sort_index().to_dict()

    montage_path = report_root / "figures" / "preprocess_samples.png"
    save_montage(output_root / "train", montage_path)
    summary = {
        "tag": args.tag,
        "crop_frac": args.crop_frac,
        "size": list(IMG_SIZE),
        "resample": "LANCZOS",
        "counts": counts,
        "class_counts": class_counts,
        "deterministic": True,
        "montage": montage_path.relative_to(base_dir).as_posix(),
    }
    summary_path = output_root / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nHASIL PREPROCESSING")
    for split in SPLITS:
        print(f"{split.capitalize():12}: {counts[split]}")
    print(f"Total        : {sum(counts.values())}")
    print(f"Determinisme : lulus")
    print(f"Montase      : {montage_path}")
    print(f"Summary      : {summary_path}")


if __name__ == "__main__":
    main()
