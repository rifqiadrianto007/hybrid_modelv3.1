from __future__ import annotations

import hashlib
import imagehash
import pandas as pd
from pathlib import Path
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
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "splits"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
DEDUP_THRESHOLD = 8
REPORT_COLUMNS = [
    "dropped_path", "dropped_label", "counterpart_path", "counterpart_label",
    "distance", "reason",
]
OUTPUT_COLUMNS = ["image_id", "filepath", "label", "source", "split"]


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())

def relative_path(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()

def collect_images(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = base_dir / "data" / "raw"
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        relative_raw = path.relative_to(raw_dir)
        if len(relative_raw.parts) < 2:
            skipped.append({"filepath": relative_path(path, base_dir), "reason": "invalid_structure"})
            continue
        source = SOURCE_MAP.get(normalize_text(relative_raw.parts[0]))
        label = LABEL_MAP.get(normalize_text(relative_raw.parts[1]))
        if source is None or label is None:
            skipped.append({"filepath": relative_path(path, base_dir), "reason": "unmapped_source_or_label"})
            continue
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as error:
            skipped.append({
                "filepath": relative_path(path, base_dir),
                "reason": f"unreadable_image: {type(error).__name__}",
            })
            continue
        rows.append({
            "filepath": relative_path(path, base_dir),
            "absolute_path": str(path),
            "label": label,
            "source": source,
        })
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
    records = []
    for row in data.itertuples(index=False):
        md5, phash, variants = calculate_hashes(Path(row.absolute_path))
        records.append({**row._asdict(), "md5": md5, "phash": phash, "variants": variants})
    result = pd.DataFrame(records)
    result["_phash_int"] = result["phash"].map(lambda value: int(value, 16))
    result["_variants_int"] = result["variants"].map(lambda values: tuple(int(value, 16) for value in values))
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

def deduplicate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if data.empty:
        summary = {"scanned": 0, "same_label_removed": 0, "cross_label_losers_removed": 0,
                   "ambiguous_removed": 0, "final_total": 0}
        return data, pd.DataFrame(columns=REPORT_COLUMNS), summary

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
                (phash_values[right_index] ^ candidate).bit_count()
                for candidate in variant_values[left_index]
            )
            if distance > DEDUP_THRESHOLD:
                continue
            if labels[left_index] == labels[right_index]:
                union_find.union(left_index, right_index)
            else:
                cross_label_pairs.append((left_index, right_index, distance))

    groups: dict[int, list[int]] = {}
    for index in range(len(data)):
        groups.setdefault(union_find.find(index), []).append(index)
    root_by_index = {index: root for root, members in groups.items() for index in members}
    source_priority = {"primer": 0, "sekunder": 1}
    representatives = {
        root: min(members, key=lambda index: (
            source_priority[data.iloc[index]["source"]], data.iloc[index]["filepath"]
        ))
        for root, members in groups.items()
    }
    contains_primer = {
        root: any(data.iloc[index]["source"] == "primer" for index in members)
        for root, members in groups.items()
    }
    cluster_reasons: dict[int, str] = {}
    conflicts: dict[int, list[tuple[int, int, int, str]]] = {root: [] for root in groups}
    for left_index, right_index, distance in cross_label_pairs:
        left_root, right_root = root_by_index[left_index], root_by_index[right_index]
        if left_root == right_root:
            continue
        left_has_primer, right_has_primer = contains_primer[left_root], contains_primer[right_root]
        if left_has_primer != right_has_primer:
            reason = "cross_label_loser"
            loser_root = right_root if left_has_primer else left_root
            cluster_reasons.setdefault(loser_root, reason)
        else:
            reason = "cross_label_ambiguous"
            cluster_reasons[left_root] = reason
            cluster_reasons[right_root] = reason
        conflicts[left_root].append((left_index, right_index, distance, reason))
        conflicts[right_root].append((left_index, right_index, distance, reason))

    reports: list[dict[str, object]] = []
    same_label_removed = 0
    for root, members in groups.items():
        representative = representatives[root]
        if root in cluster_reasons:
            continue
        for dropped in members:
            if dropped == representative:
                continue
            same_label_removed += 1
            reports.append({
                "dropped_path": data.iloc[dropped]["filepath"],
                "dropped_label": data.iloc[dropped]["label"],
                "counterpart_path": data.iloc[representative]["filepath"],
                "counterpart_label": data.iloc[representative]["label"],
                "distance": pair_distance(data.iloc[dropped], data.iloc[representative]),
                "reason": "same_label_duplicate",
            })

    for root, reason in cluster_reasons.items():
        root_conflicts = [conflict for conflict in conflicts[root] if conflict[3] == reason]
        for member in groups[root]:
            conflict = root_conflicts[0]
            left_index, right_index, distance, _ = conflict
            counterpart = right_index if root_by_index[left_index] == root else left_index
            reports.append({
                "dropped_path": data.iloc[member]["filepath"],
                "dropped_label": data.iloc[member]["label"],
                "counterpart_path": data.iloc[counterpart]["filepath"],
                "counterpart_label": data.iloc[counterpart]["label"],
                "distance": distance,
                "reason": reason,
            })

    kept_indices = [
        representatives[root] for root in groups if root not in cluster_reasons
    ]
    loser_roots = {
        root for root, reason in cluster_reasons.items() if reason == "cross_label_loser"
    }
    ambiguous_roots = {
        root for root, reason in cluster_reasons.items() if reason == "cross_label_ambiguous"
    }
    removed_losers = sum(len(groups[root]) for root in loser_roots)
    removed_ambiguous = sum(len(groups[root]) for root in ambiguous_roots)
    summary = {
        "scanned": len(data),
        "same_label_removed": same_label_removed,
        "cross_label_losers_removed": removed_losers,
        "ambiguous_removed": removed_ambiguous,
        "final_total": len(kept_indices),
    }
    return data.iloc[sorted(kept_indices)].copy().reset_index(drop=True), pd.DataFrame(reports, columns=REPORT_COLUMNS), summary

def stratified_split(
    data: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = data["label"] + "|" + data["source"]
    stratify = key
    if key.value_counts().min() < 8:
        stratify = data["label"]
    try:
        return train_test_split(data, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(data, test_size=test_size, random_state=seed, stratify=data["label"])

def add_image_ids(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["image_id"] = data["filepath"].map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())
    return data

def print_table(title: str, table: pd.DataFrame) -> None:
    print(title)
    print(table.round(4).to_string())

def main() -> None:
    ratios = [TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO]
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, skipped = collect_images(BASE_DIR)
    skipped_path = OUTPUT_DIR / "skipped.csv"
    if skipped.empty:
        skipped_path.unlink(missing_ok=True)
    else:
        skipped.to_csv(skipped_path, index=False)
    if data.empty:
        raise ValueError("No valid images found in data/raw/{primer,sekunder}/{class folder}.")

    data = hash_images(data)
    data, dedup_report, dedup_summary = deduplicate(data)
    dedup_report_path = OUTPUT_DIR / "dedup_report.csv"
    if dedup_report.empty:
        dedup_report_path.unlink(missing_ok=True)
    else:
        dedup_report.to_csv(dedup_report_path, index=False)
    print_table("Dedup summary", pd.DataFrame([dedup_summary]))

    data = add_image_ids(data)
    train_df, remainder_df = stratified_split(data, VALIDATION_RATIO + TEST_RATIO, SEED)
    validation_fraction = TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    validation_df, test_df = stratified_split(remainder_df, validation_fraction, SEED)
    split_frames = {"train": train_df, "validation": validation_df, "test": test_df}
    for split_name, split_df in split_frames.items():
        split_df = split_df.copy()
        split_df["split"] = split_name
        split_frames[split_name] = split_df
        split_df[OUTPUT_COLUMNS].to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)
    all_data = pd.concat(split_frames.values(), ignore_index=True)
    all_data[OUTPUT_COLUMNS].to_csv(OUTPUT_DIR / "all_data.csv", index=False)

    image_ids = [set(frame["image_id"]) for frame in split_frames.values()]
    filepaths = [set(frame["filepath"]) for frame in split_frames.values()]
    assert not (image_ids[0] & image_ids[1] or image_ids[0] & image_ids[2] or image_ids[1] & image_ids[2])
    assert not (filepaths[0] & filepaths[1] or filepaths[0] & filepaths[2] or filepaths[1] & filepaths[2])
    for frame in split_frames.values():
        assert set(CLASS_NAMES).issubset(set(frame["label"]))

    class_split = pd.crosstab(all_data["label"], all_data["split"]).reindex(
        index=CLASS_NAMES, columns=["train", "validation", "test"], fill_value=0
    )
    source_split = pd.crosstab(all_data["source"], all_data["split"]).reindex(
        index=list(SOURCE_MAP.values()), columns=["train", "validation", "test"], fill_value=0
    )
    print_table("Class by split", class_split)
    print_table("Source by split", source_split)
    print_table("Dataset sizes", pd.DataFrame([{
        "total": len(all_data),
        "train": len(train_df),
        "validation": len(validation_df),
        "test": len(test_df),
    }]))
    if skipped.shape[0] or dedup_summary["ambiguous_removed"]:
        print(f"WARNING: {len(skipped)} skipped file(s); {dedup_summary['ambiguous_removed']} ambiguous image(s) removed.")

if __name__ == "__main__":
    main()
