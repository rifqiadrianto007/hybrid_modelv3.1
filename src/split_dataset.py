from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import imagehash
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_MAP = {"primer": "primer", "sekunder": "sekunder"}
LABEL_MAP = {
    "leaf curl": "leaf_curl",
    "leaf spot": "leaf_spot",
    "yellowish": "yellowish",
    "healthy leaf": "healthy_leaf",
}
CLASS_NAMES = ["leaf_curl", "leaf_spot", "yellowish", "healthy_leaf"]


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())

def relative_path(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()

def collect_images(base_dir: Path, warnings_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = base_dir / "data" / "raw"
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        rel_raw = path.relative_to(raw_dir)
        if len(rel_raw.parts) < 2:
            skipped.append({"filepath": relative_path(path, base_dir), "reason": "invalid_structure"})
            continue
        source = SOURCE_MAP.get(normalize_text(rel_raw.parts[0]))
        label = LABEL_MAP.get(normalize_text(rel_raw.parts[1]))
        if source is None or label is None:
            skipped.append({"filepath": relative_path(path, base_dir), "reason": "unmapped_source_or_label"})
            continue
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            skipped.append({"filepath": relative_path(path, base_dir), "reason": f"unreadable_image: {type(exc).__name__}"})
            continue
        rows.append({
            "filepath": relative_path(path, base_dir),
            "absolute_path": str(path),
            "label": label,
            "source": source,
        })
    if skipped:
        warnings_list.append(f"{len(skipped)} file(s) masuk skipped.csv.")
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["filepath", "reason"])

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def calculate_hashes(path: Path) -> tuple[str, str, list[str]]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        variants = []
        for angle in (0, 90, 180, 270):
            rotated = image.rotate(angle, expand=True)
            for flipped in (False, True):
                variant = ImageOps.mirror(rotated) if flipped else rotated
                variants.append(str(imagehash.phash(variant)))
        return md5_file(path), variants[0], variants

def hash_images(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        path = Path(row.absolute_path)
        md5, phash, variants = calculate_hashes(path)
        records.append({**row._asdict(), "md5": md5, "phash": phash, "variants": variants})
    result = pd.DataFrame(records)
    result["_phash_int"] = result["phash"].map(lambda value: int(value, 16))
    result["_variants_int"] = result["variants"].map(lambda value: tuple(int(item, 16) for item in value))
    return result

class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

def pair_distance(first: pd.Series, second: pd.Series) -> int:
    second_hash = second["_phash_int"]
    return min((second_hash ^ candidate).bit_count() for candidate in first["_variants_int"])

def deduplicate(data: pd.DataFrame, threshold: int, warnings_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    report_columns = ["dropped_path", "kept_path", "distance", "note"]
    if data.empty:
        return data, pd.DataFrame(columns=report_columns), {"clusters": 0, "dropped": 0, "largest_cluster": 0}
    data = data.reset_index(drop=True)
    union_find = UnionFind(len(data))
    cross_label_pairs: list[tuple[int, int, int]] = []
    labels = data["label"].tolist()
    md5_values = data["md5"].tolist()
    phash_values = data["_phash_int"].tolist()
    variant_values = data["_variants_int"].tolist()
    for left_index in range(len(data)):
        for right_index in range(left_index + 1, len(data)):
            distance = 0 if md5_values[left_index] == md5_values[right_index] else min(
                (phash_values[right_index] ^ candidate).bit_count() for candidate in variant_values[left_index]
            )
            if distance > threshold:
                continue
            if labels[left_index] == labels[right_index]:
                union_find.union(left_index, right_index)
            else:
                cross_label_pairs.append((left_index, right_index, distance))
    groups: dict[int, list[int]] = {}
    for index in range(len(data)):
        groups.setdefault(union_find.find(index), []).append(index)
    reports: list[dict[str, object]] = []
    kept_indices: set[int] = set()
    source_priority = {"primer": 0, "sekunder": 1}
    for members in groups.values():
        representative = min(members, key=lambda index: (source_priority[data.iloc[index]["source"]], data.iloc[index]["filepath"]))
        kept_indices.add(representative)
        for dropped in members:
            if dropped != representative:
                reports.append({
                    "dropped_path": data.iloc[dropped]["filepath"],
                    "kept_path": data.iloc[representative]["filepath"],
                    "distance": pair_distance(data.iloc[dropped], data.iloc[representative]),
                    "note": "duplicate",
                })
    for left_index, right_index, distance in cross_label_pairs:
        reports.append({
            "dropped_path": data.iloc[left_index]["filepath"],
            "kept_path": data.iloc[right_index]["filepath"],
            "distance": distance,
            "note": "cross_label",
        })
    if cross_label_pairs:
        warnings_list.append(f"{len(cross_label_pairs)} WARNING: near-duplicate lintas label terdeteksi (indikasi label salah).")
    cluster_sizes = [len(members) for members in groups.values()]
    largest_cluster = max(cluster_sizes, default=0)
    if largest_cluster > 5:
        warnings_list.append(f"WARNING: ukuran klaster duplikat terbesar adalah {largest_cluster} (> 5).")
    summary = {"clusters": sum(size > 1 for size in cluster_sizes), "dropped": len(data) - len(kept_indices), "largest_cluster": largest_cluster}
    return data.iloc[sorted(kept_indices)].copy().reset_index(drop=True), pd.DataFrame(reports, columns=report_columns), summary

def allocate_per_class(data: pd.DataFrame, count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_groups: list[pd.DataFrame] = []
    quota_rows: list[dict[str, object]] = []
    sources = list(SOURCE_MAP.values())
    for class_index, label in enumerate(CLASS_NAMES):
        class_data = data[data["label"] == label]
        if len(class_data) < count:
            raise ValueError(f"Kelas {label} hanya memiliki {len(class_data)} gambar, kurang dari --per-class {count}.")
        source_counts = class_data["source"].value_counts().to_dict()
        desired = {source: count * source_counts.get(source, 0) / len(class_data) for source in sources}
        quotas = {source: min(int(desired[source]), source_counts.get(source, 0)) for source in sources}
        remaining = count - sum(quotas.values())
        while remaining:
            candidates = [source for source in sources if quotas[source] < source_counts.get(source, 0)]
            if not candidates:
                raise ValueError(f"Tidak dapat mengalokasikan --per-class untuk kelas {label}.")
            source = max(candidates, key=lambda item: (desired[item] - int(desired[item]), -sources.index(item)))
            quotas[source] += 1
            remaining -= 1
        for source in sources:
            source_data = class_data[class_data["source"] == source]
            quota = quotas[source]
            if quota:
                selected_groups.append(source_data.sample(n=quota, random_state=seed + class_index))
            quota_rows.append({"label": label, "source": source, "quota": quota, "available": len(source_data)})
    return pd.concat(selected_groups, ignore_index=True), pd.DataFrame(quota_rows)

def stratified_split(data: pd.DataFrame, test_size: float, seed: int, name: str, warnings_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = data["label"] + "|" + data["source"]
    stratify = key
    if key.value_counts().min() < 8:
        warnings_list.append(f"WARNING: stratum label|source pada tahap {name} memiliki anggota < 8; fallback ke label.")
        stratify = data["label"]
    try:
        return train_test_split(data, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError as exc:
        warnings_list.append(f"WARNING: stratifikasi label|source gagal pada tahap {name}; fallback ke label ({exc}).")
        return train_test_split(data, test_size=test_size, random_state=seed, stratify=data["label"])

def add_image_ids(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["image_id"] = data["filepath"].map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())
    return data

def main() -> None:
    parser = argparse.ArgumentParser(description="Membuat manifest split dataset tanpa menyalin gambar.")
    parser.add_argument("--tag", choices=["pilot"], required=True)
    parser.add_argument("--per-class", type=int, default=None)

    args = parser.parse_args()
    if args.per_class is not None and args.per_class <= 0:
        raise ValueError("--per-class harus lebih besar dari nol.")
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train, validation, dan test ratio harus positif dan berjumlah 1.")
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "splits" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []

    data, skipped = collect_images(base_dir, warnings_list)
    skipped_path = output_dir / "skipped.csv"
    if skipped.empty:
        skipped_path.unlink(missing_ok=True)
    else:
        skipped.to_csv(skipped_path, index=False)
    if data.empty:
        raise ValueError("Tidak ditemukan gambar valid dalam struktur data/raw/{Primer,Sekunder}/{kelas}.")
    data = hash_images(data)
    data, dedup_report, dedup_summary = deduplicate(data, args.dedup_threshold, warnings_list)
    dedup_report_path = output_dir / "dedup_report.csv"
    if dedup_report.empty:
        dedup_report_path.unlink(missing_ok=True)
    else:
        dedup_report.to_csv(dedup_report_path, index=False)
    print(f"Dedup: {dedup_summary['clusters']} klaster, {dedup_summary['dropped']} dibuang, klaster terbesar {dedup_summary['largest_cluster']}")

    if args.per_class is not None:
        data, quota_table = allocate_per_class(data, args.per_class, args.seed)
        print("Jatah kelas x source:")
        print(quota_table.to_string(index=False))

    data = add_image_ids(data)
    train_df, remainder_df = stratified_split(data, args.val_ratio + args.test_ratio, args.seed, "train-vs-sisa", warnings_list)
    val_fraction = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = stratified_split(remainder_df, val_fraction, args.seed, "validation-vs-test", warnings_list)
    split_frames = {"train": train_df, "validation": val_df, "test": test_df}
    for split_name, split_df in split_frames.items():
        split_df = split_df.copy()
        split_df["split"] = split_name
        split_frames[split_name] = split_df
        split_df[["image_id", "filepath", "label", "source", "split"]].to_csv(output_dir / f"{split_name}.csv", index=False)
    all_data = pd.concat(split_frames.values(), ignore_index=True)
    all_data[["image_id", "filepath", "label", "source", "split"]].to_csv(output_dir / "all_data.csv", index=False)

    image_ids = [set(frame["image_id"]) for frame in split_frames.values()]
    filepaths = [set(frame["filepath"]) for frame in split_frames.values()]
    assert not (image_ids[0] & image_ids[1] or image_ids[0] & image_ids[2] or image_ids[1] & image_ids[2])
    assert not (filepaths[0] & filepaths[1] or filepaths[0] & filepaths[2] or filepaths[1] & filepaths[2])
    for frame in split_frames.values():
        assert set(CLASS_NAMES).issubset(set(frame["label"]))

    class_split = pd.crosstab(all_data["label"], all_data["split"]).reindex(index=CLASS_NAMES, columns=["train", "validation", "test"], fill_value=0)
    source_split = pd.crosstab(all_data["source"], all_data["split"]).reindex(index=list(SOURCE_MAP.values()), columns=["train", "validation", "test"], fill_value=0)
    print("Kelas x split:")
    print(class_split.to_string())
    print("Source x split:")
    print(source_split.to_string())
    print(f"Skipped: {len(skipped)}")
    print(f"Total: {len(all_data)} | train={len(train_df)} validation={len(val_df)} test={len(test_df)}")
    if warnings_list:
        print("WARNING:")
        for warning in warnings_list:
            print(f"- {warning}")

if __name__ == "__main__":
    main()
