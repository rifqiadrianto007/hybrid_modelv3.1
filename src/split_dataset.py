# FOLDER PRIMER DAN SEKUNDER

# Membagi dataset menjadi folder train/validation/test dengan proporsi 70:15:15
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mengatur ulang seed untuk memastikan hasil yang konsisten
SEED = 42
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Normalisasi teks untuk pencocokan label
def normalize_text(text: str) -> str :
    # Hapus spasi, lowercase, dan ganti karakter pemisah dengan spasi
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())

def infer_label_from_path(path_obj : Path) -> str | None :
    # Normalisasi setiap path dengan nama kelas yang sesuai
    parts_norm = [normalize_text(part) for part in path_obj.parts]
    class_names_norm = {normalize_text(name): name for name in CLASS_NAMES}
    for part in parts_norm :
        if part in class_names_norm :
            return class_names_norm[part]
    return None

# Fungsi untuk mengumpulkan gambar dari folder raw dan membuat DataFrame
def collect_images(root_dir : Path, source_name : str) -> pd.DataFrame :
    rows = []
    # Mengatasi jika folder kosong
    if not root_dir.exists() :
        return pd.DataFrame(columns=["filepath", "label", "source"])
    # Memeriksa setiap file dalam folder raw
    for file_path in root_dir.rglob("*") :
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXT :
            continue
        # Mengumpulkan label dari setiap path
        label = infer_label_from_path(file_path)
        if label is None :
            continue
        # Menambahkan baris ke DataFrame
        rows.append(
            {
                "filepath": str(file_path),
                "label": label,
                "source": source_name
            }
        )
    # DataFrame berisi semua gambar dengan label dan sumber
    return pd.DataFrame(rows)

# Membuat nama file baru setelah di split
def safe_filename(source : str, class_name : str, original_path : Path) -> str :
    stem = original_path.stem.replace(" ", "_") # Spasi menjadi underscore
    stem = stem.replace("/", "_").replace("\\", "_") # Pemisah menjadi underscore
    # Format nama file: source__class_name__original_filename.ext
    return f"{source}__{class_name.replace(' ', '_')}__{stem}{original_path.suffix.lower()}"

# Menyalin ke folder split yang sesuai dengan kelas file baru
def copy_split(split_df : pd.DataFrame, split_dir : Path) -> None :
    for _, row in split_df.iterrows() :
        src_path = Path(row["filepath"])
        class_dir = split_dir / row["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_path = class_dir / safe_filename(row["source"], row["label"], src_path)
        shutil.copy2(src_path, dst_path)

# Fungsi utama untuk menjalankan proses split dataset
def main() -> None :
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type = str, default = None) # Root project opsional
    parser.add_argument("--train-ratio", type = float, default = 0.70) # 70% train
    parser.add_argument("--val-ratio", type = float, default = 0.15) # 15% validation
    parser.add_argument("--test-ratio", type = float, default = 0.15) # 15% test
    args = parser.parse_args()
    # Menentukan folder asli dan folder hasil split
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    raw_primer = base_dir / "data" / "raw" / "primer" # Folder asli primer
    raw_sekunder = base_dir / "data" / "raw" / "sekunder" # Folder asli sekunder
    split_root = base_dir / "data" / "splits" # Folder hasil split
    split_root.mkdir(parents = True, exist_ok=True) # Membuat folder spilt jika belum ada
    # DataFrame gabungan primer dan sekunder hasil split
    primer_df = collect_images(raw_primer, "primer")
    sekunder_df = collect_images(raw_sekunder, "sekunder")
    data_df = pd.concat([primer_df, sekunder_df], ignore_index = True)
    # Validasi dataset dan proporsi split
    if data_df.empty :
        raise ValueError("Gagal memuat dataset. Tidak ditemukan gambar valid di folder raw.")
    # Memeriksa proporsi split (70:15:15)
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0) :
        raise ValueError("Proporsi split tidak sesuai.")
    
    train_df, temp_df = train_test_split(
        data_df,
        test_size = 0.3, # 30% untuk validation + test
        random_state = SEED, # Seed untuk konsistensi
        stratify = data_df["label"]
    )
    # Menyesuaikan proporsi validation dan test dari sisa 30%
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size = (1.0 - val_ratio_adjusted),
        random_state = SEED,
        stratify=temp_df["label"]
    )
    # Membuat folder split dan menyalin gambar ke folder yang sesuai dengan kelasnya
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)] :
        # Handling jika ada/tidak folder split
        split_dir = split_root / split_name
        if split_dir.exists() :
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents = True, exist_ok = True)
        copy_split(split_df, split_dir) #
        split_df.to_csv(split_root / f"{split_name}.csv", index = False)
    # Menyimpan dataframe gabungan ke format .csv
    data_df.to_csv(split_root / "all_data.csv", index = False)
    # Mengembalikan semua hasil split
    print("Split completed")
    print(f"Total : {len(data_df)}")
    print(f"Train : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test : {len(test_df)}")
    print(f"Saved under : {split_root}")

if __name__ == "__main__" :
    main()