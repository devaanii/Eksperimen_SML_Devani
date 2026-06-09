# Skrip otomatisasi preprocessing dataset Wine Quality (White Wine).
# Menyediakan fungsi siap-pakai untuk membaca data mentah, membersihkannya,
# dan menyimpan hasil yang siap dilatih.
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final, Union

import numpy as np
import pandas as pd

# Konstanta & konfigurasi path

DATASET_NAME: Final[str] = "wine-quality-white"
RAW_DATASET_NAME: Final[str] = f"{DATASET_NAME}_raw"
PROCESSED_DATASET_NAME: Final[str] = f"{DATASET_NAME}_preprocessing"

TARGET_COLUMN: Final[str] = "quality"

_HERE: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _HERE.parent
DEFAULT_RAW_PATH: Final[Path] = _REPO_ROOT / f"{RAW_DATASET_NAME}.csv"
DEFAULT_OUTPUT_PATH: Final[Path] = _HERE / f"{PROCESSED_DATASET_NAME}.csv"

RawInput = Union[str, "os.PathLike[str]", pd.DataFrame]


class PreprocessingError(RuntimeError):
    """Kesalahan ketika data mentah gagal dibaca atau hasil gagal disimpan."""


# Langkah-langkah transformasi
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Paksa seluruh kolom menjadi numerik; nilai tak valid menjadi NaN."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah target jadi label biner: 1 bila quality >= 6, selain itu 0."""
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN].fillna(0) >= 6).astype(int)
    return df


def _impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Isi nilai hilang tiap fitur dengan median kolom yang bersangkutan."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    for col in feature_cols:
        median = df[col].median()
        if pd.isna(median):
            median = 0.0
        df[col] = df[col].fillna(median)
    return df


def _drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Buang baris duplikat lalu susun ulang indeks dari nol."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Hasilkan salinan df yang siap dilatih: seluruh kolom numerik, tanpa nilai hilang, dan target sudah biner.

    Urutan tahap (penting, menentukan hasil akhir):
      1. konversi numerik
      2. binarisasi target
      3. imputasi median pada fitur
      4. buang duplikat & reset indeks
      5. pastikan kembali seluruh kolom numerik
    """
    out = df.copy()
    out = _coerce_numeric(out)
    out = _binarize_target(out)
    out = _impute_median(out)
    out = _drop_duplicate_rows(out)
    out = out.apply(pd.to_numeric)
    return out


# Baca / tulis data

def load_dataset(path: Union[str, "os.PathLike[str]"]) -> pd.DataFrame:
    """Baca berkas CSV mentah (pemisah titik-koma, dengan baris header)."""
    try:
        return pd.read_csv(path, sep=";")
    except Exception as exc:
        raise PreprocessingError(
            f"Tidak dapat membaca dataset mentah dari {os.fspath(path)!r}: {exc}"
        ) from exc


def save_dataset(df: pd.DataFrame, path: Union[str, "os.PathLike[str]"]) -> Path:
    """Simpan df ke path secara aman (tulis ke berkas sementara lalu ganti nama)."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise PreprocessingError(
            f"Gagal membuat folder tujuan untuk {os.fspath(path)!r}: {exc}"
        ) from exc

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise PreprocessingError(
            f"Gagal menyimpan dataset hasil preprocessing ke {os.fspath(path)!r}: {exc}"
        ) from exc
    return target


def preprocess_dataset(
    raw_input: RawInput,
    output_path: Union[str, "os.PathLike[str]", None] = None,
) -> pd.DataFrame:
    """Muat data mentah, terapkan transformasi, simpan bila diminta, lalu kembalikan hasilnya."""
    raw = raw_input if isinstance(raw_input, pd.DataFrame) else load_dataset(raw_input)

    processed = clean_dataset(raw)

    if output_path is not None:
        save_dataset(processed, output_path)

    return processed


def _print_summary(processed: pd.DataFrame, raw_path, output_path) -> None:
    """Cetak ringkasan singkat hasil preprocessing ke konsol."""
    missing_total = int(processed.isna().sum().sum())
    non_numeric = [
        col
        for col in processed.columns
        if not pd.api.types.is_numeric_dtype(processed[col])
    ]
    print(f"Berkas mentah        : {os.fspath(raw_path)}")
    print(f"Hasil disimpan ke    : {os.fspath(output_path)}")
    print(f"Dimensi akhir        : {processed.shape[0]} baris x {processed.shape[1]} kolom")
    print(f"Sisa nilai hilang    : {missing_total}")
    print(f"Kolom non-numerik    : {non_numeric if non_numeric else 'tidak ada'}")
    if TARGET_COLUMN in processed.columns:
        counts = processed[TARGET_COLUMN].value_counts().to_dict()
        print(f"Komposisi target     : {counts}")


def main(
    raw_path: Union[str, "os.PathLike[str]"] = DEFAULT_RAW_PATH,
    output_path: Union[str, "os.PathLike[str]"] = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Jalankan pipeline penuh: baca data mentah, simpan hasil olahan, cetak ringkasannya."""
    processed = preprocess_dataset(raw_path, output_path)
    _print_summary(processed, raw_path, output_path)
    return processed


if __name__ == "__main__":
    main()
