"""Inspect a folder containing one .mat file and one .csv file."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def find_single_file(folder: Path, suffix: str) -> Path:
    matches = list(folder.glob(f"*{suffix}"))
    if len(matches) == 0:
        sys.exit(f"Error: no {suffix} file found in {folder}")
    if len(matches) > 1:
        sys.exit(f"Error: multiple {suffix} files found in {folder}: {matches}")
    return matches[0]


def inspect_mat(mat_path: Path) -> dict:
    print(f"=== MAT file: {mat_path.name} ===")
    data = {}
    with h5py.File(str(mat_path), "r") as f:
        for key in sorted(f.keys()):
            ds = f[key]
            if isinstance(ds, h5py.Dataset):
                print(f"  {key:30s}  shape={str(ds.shape):20s}  dtype={ds.dtype}")
                data[key] = ds.shape
            elif isinstance(ds, h5py.Group):
                print(f"  {key:30s}  [group with {len(ds)} members]")
            else:
                print(f"  {key:30s}  type={type(ds).__name__}")
    return data


def inspect_csv(csv_path: Path) -> pd.DataFrame:
    print(f"\n=== CSV file: {csv_path.name} ===")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


def check_alignment(mat_shapes: dict, df: pd.DataFrame) -> None:
    print("\n=== Alignment check ===")
    key = "normTraceMatrix"
    if key not in mat_shapes:
        print(f"  '{key}' not found in .mat file — skipping check.")
        return
    shape = mat_shapes[key]
    mat_dim = shape[1]
    csv_rows = df.shape[0]
    if mat_dim == csv_rows:
        print(f"  OK: CSV rows ({csv_rows}) == normTraceMatrix second dim ({mat_dim})")
    else:
        print(f"  MISMATCH: CSV rows ({csv_rows}) != normTraceMatrix second dim ({mat_dim})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .mat / .csv data folder.")
    parser.add_argument("folder", type=Path, help="Path to folder with one .mat and one .csv")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        sys.exit(f"Error: {folder} is not a directory")

    mat_path = find_single_file(folder, ".mat")
    csv_path = find_single_file(folder, ".csv")

    mat_data = inspect_mat(mat_path)
    df = inspect_csv(csv_path)
    check_alignment(mat_data, df)


if __name__ == "__main__":
    main()
