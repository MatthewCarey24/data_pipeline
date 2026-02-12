"""Extract a single-neuron sample from a loaded .mat dict and metadata DataFrame,
and pack / unpack lists of samples as WebDataset .tar shards."""

import io
import json
import tarfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-neuron extraction
# ---------------------------------------------------------------------------

def get_neuron_sample(mat_data: dict, metadata_df: pd.DataFrame, i: int) -> dict:
    """Return trace, stimulus, and metadata for neuron *i*.

    Parameters
    ----------
    mat_data : dict
        Already-loaded .mat contents (e.g. from ``h5py.File``).
        Must contain keys ``"normTraceMatrix"`` and ``"stimulusCommand"``.
    metadata_df : pd.DataFrame
        CSV metadata with one row per neuron.
    i : int
        Neuron (column) index.

    Returns
    -------
    dict with keys:
        trace     - float32 numpy array, shape (11481,)
        stimulus  - float32 numpy array, shape (11481,)
        metadata  - dict of all CSV column values for row *i*
    """
    trace = np.asarray(mat_data["normTraceMatrix"][:, i], dtype=np.float32)
    stimulus = np.asarray(mat_data["stimulusCommand"], dtype=np.float32).squeeze()
    metadata = metadata_df.iloc[i].to_dict()
    return {"trace": trace, "stimulus": stimulus, "metadata": metadata}


# ---------------------------------------------------------------------------
# WebDataset tar writing / reading
# ---------------------------------------------------------------------------

def _npy_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to .npy bytes in memory."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _add_tar_entry(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    """Append an in-memory bytes blob as a file inside an open TarFile."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def write_tar_shard(samples: list[dict], tar_path: str) -> None:
    """Write a list of sample dicts to a single .tar shard.

    Each sample is stored as three entries:
        {index}.trace.npy      - float32 numpy array
        {index}.stimulus.npy   - float32 numpy array
        {index}.metadata.json  - JSON object with all CSV fields

    Parameters
    ----------
    samples : list[dict]
        Each dict must have keys ``trace``, ``stimulus``, ``metadata``
        (as returned by :func:`get_neuron_sample`).
    tar_path : str | Path
        Destination tar file path.
    """
    with tarfile.open(str(tar_path), "w") as tf:
        for idx, sample in enumerate(samples):
            key = f"{idx:06d}"
            _add_tar_entry(tf, f"{key}.trace.npy", _npy_to_bytes(sample["trace"]))
            _add_tar_entry(tf, f"{key}.stimulus.npy", _npy_to_bytes(sample["stimulus"]))
            _add_tar_entry(
                tf,
                f"{key}.metadata.json",
                json.dumps(sample["metadata"]).encode("utf-8"),
            )


def read_tar_shard(tar_path: str):
    """Yield sample dicts from a .tar shard written by :func:`write_tar_shard`.

    Yields
    ------
    dict with keys ``trace`` (ndarray), ``stimulus`` (ndarray), ``metadata`` (dict).
    """
    with tarfile.open(str(tar_path), "r") as tf:
        members = sorted(tf.getmembers(), key=lambda m: m.name)

        # Group consecutive triplets that share the same numeric prefix.
        pending: dict = {}
        current_key: str | None = None

        for member in members:
            # e.g. "000000.trace.npy" -> key="000000", rest="trace.npy"
            key, rest = member.name.split(".", maxsplit=1)

            if current_key is not None and key != current_key:
                # Emit the completed sample before starting the next one.
                yield _assemble_sample(pending)
                pending = {}

            current_key = key
            raw = tf.extractfile(member).read()

            if rest == "trace.npy":
                pending["trace"] = np.load(io.BytesIO(raw))
            elif rest == "stimulus.npy":
                pending["stimulus"] = np.load(io.BytesIO(raw))
            elif rest == "metadata.json":
                pending["metadata"] = json.loads(raw)

        # Emit the last sample.
        if pending:
            yield _assemble_sample(pending)


def _assemble_sample(parts: dict) -> dict:
    """Validate and return a sample dict from its parsed parts."""
    assert "trace" in parts and "stimulus" in parts and "metadata" in parts, (
        f"Incomplete sample, got keys: {list(parts.keys())}"
    )
    return parts


# ---------------------------------------------------------------------------
# Full-file conversion: .mat + .csv  ->  shuffled tar shards
# ---------------------------------------------------------------------------

def convert_mat_csv_pair(
    mat_path: str,
    csv_path: str,
    out_dir: str,
    shard_size: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Load a .mat / .csv pair and write shuffled WebDataset tar shards.

    Parameters
    ----------
    mat_path : str | Path
        Path to the HDF5-based .mat file.
    csv_path : str | Path
        Path to the matching metadata CSV.
    out_dir : str | Path
        Directory where shard .tar files will be written (created if needed).
    shard_size : int
        Maximum number of samples per shard.
    seed : int
        RNG seed for reproducible shuffling.

    Returns
    -------
    list[str]
        Paths to all shard files written.
    """
    import h5py, os, math
    from pathlib import Path

    mat_path = Path(mat_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_stem = mat_path.stem  # e.g. "DFP0037_traceMatrix"

    # -- load data -------------------------------------------------------------
    print(f"Loading {mat_path.name} ...")
    with h5py.File(str(mat_path), "r") as f:
        mat_data = {k: f[k][:] for k in f.keys()}
    metadata_df = pd.read_csv(csv_path)

    n_neurons = mat_data["normTraceMatrix"].shape[1]
    n_shards = math.ceil(n_neurons / shard_size)
    print(f"  {n_neurons} neurons -> {n_shards} shards of up to {shard_size}")

    # -- shuffle neuron indices ------------------------------------------------
    rng = np.random.default_rng(seed)
    indices = np.arange(n_neurons)
    rng.shuffle(indices)

    # -- iterate in chunks, extract samples, write shards ----------------------
    shard_paths: list[str] = []
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n_neurons)
        chunk_indices = indices[start:end]

        samples = [get_neuron_sample(mat_data, metadata_df, int(i)) for i in chunk_indices]

        shard_name = f"{mat_stem}_{shard_idx:04d}.tar"
        shard_path = out_dir / shard_name
        write_tar_shard(samples, shard_path)
        shard_paths.append(str(shard_path))

        size_kb = shard_path.stat().st_size / 1024
        print(f"  [{shard_idx + 1}/{n_shards}] {shard_name}  "
              f"({len(samples)} samples, {size_kb:.0f} KB)")

    print(f"Done — {n_shards} shards in {out_dir}")
    return shard_paths


# ---------------------------------------------------------------------------
# Directory-level conversion: scan, pair, convert all
# ---------------------------------------------------------------------------

def convert_directory(
    input_dir: str,
    output_dir: str,
    shard_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Scan *input_dir* for .mat/.csv pairs and convert each to tar shards.

    Naming convention:
        {ID}_traceMatrix.mat  <->  {ID}_sourceMetadata.csv

    Parameters
    ----------
    input_dir : str | Path
        Directory containing .mat and .csv files.
    output_dir : str | Path
        Root output directory.  Each plate gets a sub-directory
        ``output_dir/{plate_id}/`` containing its shard .tar files.
    shard_size : int
        Max samples per shard (forwarded to :func:`convert_mat_csv_pair`).
    seed : int
        RNG seed for reproducible shuffling.

    Returns
    -------
    dict with keys:
        pairs_processed  - number of .mat/.csv pairs converted
        total_shards     - total number of shard files written
        skipped_mat      - list of .mat filenames with no matching CSV
        skipped_csv      - list of .csv filenames with no matching .mat
    """
    from pathlib import Path

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # -- discover .mat and .csv files ------------------------------------------
    mat_files = sorted(input_dir.glob("*_traceMatrix.mat"))
    csv_files = sorted(input_dir.glob("*_sourceMetadata.csv"))

    # Build lookup: plate_id -> path
    mat_by_id = {}
    for p in mat_files:
        plate_id = p.name.removesuffix("_traceMatrix.mat")
        mat_by_id[plate_id] = p

    csv_by_id = {}
    for p in csv_files:
        plate_id = p.name.removesuffix("_sourceMetadata.csv")
        csv_by_id[plate_id] = p

    paired_ids = sorted(set(mat_by_id) & set(csv_by_id))

    skipped_mat = sorted(set(mat_by_id) - set(csv_by_id))
    skipped_csv = sorted(set(csv_by_id) - set(mat_by_id))

    for plate_id in skipped_mat:
        print(f"  WARNING: skipping {mat_by_id[plate_id].name} — no matching CSV")
    for plate_id in skipped_csv:
        print(f"  WARNING: skipping {csv_by_id[plate_id].name} — no matching .mat")

    if not paired_ids:
        print(f"No matched .mat/.csv pairs found in {input_dir}")
        return {"pairs_processed": 0, "total_shards": 0,
                "skipped_mat": skipped_mat, "skipped_csv": skipped_csv}

    print(f"Found {len(paired_ids)} pair(s), "
          f"{len(skipped_mat)} .mat skipped, {len(skipped_csv)} .csv skipped\n")

    # -- convert each pair -----------------------------------------------------
    total_shards = 0
    for idx, plate_id in enumerate(paired_ids, 1):
        mat_path = mat_by_id[plate_id]
        csv_path = csv_by_id[plate_id]
        pair_out = output_dir / plate_id

        print(f"=== [{idx}/{len(paired_ids)}] {plate_id} ===")
        shard_paths = convert_mat_csv_pair(
            mat_path, csv_path, pair_out,
            shard_size=shard_size, seed=seed,
        )
        total_shards += len(shard_paths)
        print()

    # -- summary ---------------------------------------------------------------
    print("=" * 60)
    print(f"  Pairs processed: {len(paired_ids)}")
    print(f"  Total shards:    {total_shards}")
    print(f"  Skipped .mat:    {len(skipped_mat)}")
    print(f"  Skipped .csv:    {len(skipped_csv)}")
    print("=" * 60)

    return {
        "pairs_processed": len(paired_ids),
        "total_shards": total_shards,
        "skipped_mat": skipped_mat,
        "skipped_csv": skipped_csv,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Convert .mat/.csv pairs in a directory to WebDataset tar shards.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing *_traceMatrix.mat and *_sourceMetadata.csv files.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for shards (default: <input_dir>/shards).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Max samples per shard (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible shuffling (default: 42).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir / "shards"

    convert_directory(
        args.input_dir,
        output_dir,
        shard_size=args.shard_size,
        seed=args.seed,
    )
