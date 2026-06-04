# Otomatisasi preprocessing untuk Wine Quality (White Wine) dataset.
# Tujuan kode ini untuk mengubah CSV mentah menjadi data siap latih: semua kolom numerik, tanpa nilai hilang, dan targetnya dibinarisasi (quality >= 6 = good, < 6 = bad).

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final, Union

import numpy as np
import pandas as pd

# Nama dataset -> nama file artefak raw & hasil preprocessing.
DATASET_NAME: Final[str] = "wine-quality-white"
RAW_DATASET_NAME: Final[str] = f"{DATASET_NAME}_raw"
PROCESSED_DATASET_NAME: Final[str] = f"{DATASET_NAME}_preprocessing"

# Kolom target (label biner kualitas wine: good vs bad).
TARGET_COLUMN: Final[str] = "quality"

# Dataset Raw dibaca dari root repo, hasilnya ditulis di folder preprocessing ini.
_HERE: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _HERE.parent
DEFAULT_RAW_PATH: Final[Path] = _REPO_ROOT / f"{RAW_DATASET_NAME}.csv"
DEFAULT_OUTPUT_PATH: Final[Path] = _HERE / f"{PROCESSED_DATASET_NAME}.csv"

RawInput = Union[str, "os.PathLike[str]", pd.DataFrame]


class PreprocessingError(RuntimeError):
    """Galat saat data mentah gagal dibaca atau hasil gagal ditulis."""


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Kembalikan salinan df yang siap latih: semua numerik, tanpa nilai hilang, target biner."""
    out = df.copy()

    # Pastikan semua kolom numerik (konversi ke numerik bila perlu).
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Binarisasi target: quality >= 6 -> 1 (good), < 6 -> 0 (bad).
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = (out[TARGET_COLUMN].fillna(0) >= 6).astype(int)

    # Imputasi median untuk sisa nilai hilang pada kolom fitur.
    feature_cols = [c for c in out.columns if c != TARGET_COLUMN]
    for col in feature_cols:
        median = out[col].median()
        if pd.isna(median):
            median = 0.0  # kolom kosong total: jaga invarian tanpa-NaN
        out[col] = out[col].fillna(median)

    # Buang baris duplikat, lalu pastikan semua kolomnya numerik.
    out = out.drop_duplicates().reset_index(drop=True)
    out = out.apply(pd.to_numeric)

    return out


def _read_raw(path: Union[str, "os.PathLike[str]"]) -> pd.DataFrame:
    """Baca CSV mentah dengan delimiter semicolon dan header."""
    try:
        return pd.read_csv(path, sep=";")
    except Exception as exc:
        raise PreprocessingError(
            f"could not read raw dataset from {os.fspath(path)!r}: {exc}"
        ) from exc


def _write_processed(df: pd.DataFrame, path: Union[str, "os.PathLike[str]"]) -> Path:
    """Tulis df ke path secara atomik (file sementara lalu rename)."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise PreprocessingError(
            f"could not prepare output directory for {os.fspath(path)!r}: {exc}"
        ) from exc

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target)  # rename atomik ke lokasi final
    except Exception as exc:
        # Jangan tinggalkan output parsial, jadi hapus file sementara dulu.
        if tmp_path.exists():
            tmp_path.unlink()
        raise PreprocessingError(
            f"could not write processed dataset to {os.fspath(path)!r}: {exc}"
        ) from exc
    return target


def preprocess_dataset(
    raw_input: RawInput,
    output_path: Union[str, "os.PathLike[str]", None] = None,
) -> pd.DataFrame:
    """Muat data mentah, transformasikan, opsional simpan, lalu kembalikan."""
    raw = raw_input if isinstance(raw_input, pd.DataFrame) else _read_raw(raw_input)

    processed = _transform(raw)

    if output_path is not None:
        _write_processed(processed, output_path)

    return processed


def main(
    raw_path: Union[str, "os.PathLike[str]"] = DEFAULT_RAW_PATH,
    output_path: Union[str, "os.PathLike[str]"] = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Baca CSV mentah, tulis hasil preprocessing, dan cetak ringkasan singkat."""
    processed = preprocess_dataset(raw_path, output_path)

    missing_total = int(processed.isna().sum().sum())
    non_numeric = [
        col
        for col in processed.columns
        if not pd.api.types.is_numeric_dtype(processed[col])
    ]
    print(f"Read raw dataset : {os.fspath(raw_path)}")
    print(f"Wrote processed  : {os.fspath(output_path)}")
    print(f"Shape            : {processed.shape[0]} rows x {processed.shape[1]} cols")
    print(f"Missing values   : {missing_total}")
    print(f"Non-numeric cols : {non_numeric if non_numeric else 'none'}")
    if TARGET_COLUMN in processed.columns:
        counts = processed[TARGET_COLUMN].value_counts().to_dict()
        print(f"Target balance   : {counts}")
    return processed


if __name__ == "__main__":
    main()
