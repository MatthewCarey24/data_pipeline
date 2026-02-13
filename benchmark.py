"""Benchmark raw .mat/.csv loading vs pre-sharded WebDataset loading.

Measures three phases per batch:
    1. Data loading   (dataset __getitem__ + collation)
    2. Host-to-device (CPU tensor -> CUDA)
    3. Forward pass    (model inference, no backward)

Usage
-----
Raw mode (reads .mat + .csv on the fly):
    python benchmark.py raw --mat_dir /data/plates

Sharded mode (streams from .tar shards via WebDataset):
    python benchmark.py sharded --shard_dir /data/plates/shards

Common flags:
    --batch_size 32  --num_workers 4  --n_steps 50
"""

import argparse
import io
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset


FULL_TRACE_LEN = 11481  # full trace length from normTraceMatrix


# ---------------------------------------------------------------------------
# Transformer model (~200M parameters)
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """Simple encoder-only transformer sized to ~200M parameters.

    Input:  (batch, seq_len)  float32
    Output: (batch, seq_len)  float32  (dummy regression head)
    """

    def __init__(
        self,
        seq_len: int = FULL_TRACE_LEN,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 16,
        dim_feedforward: int = 4096,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN)
        x = x.unsqueeze(-1)              # (B, SEQ_LEN, 1)
        x = self.input_proj(x)           # (B, SEQ_LEN, d_model)
        x = x + self.pos_emb
        x = self.encoder(x)              # (B, SEQ_LEN, d_model)
        x = self.head(x).squeeze(-1)     # (B, SEQ_LEN)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Raw .mat/.csv dataset
# ---------------------------------------------------------------------------

class RawMatDataset(Dataset):
    """Map-style dataset that reads .mat + .csv files on the fly.

    Each __getitem__ call:
        1. Picks a random .mat file
        2. Opens it with h5py, randomly samples a neuron column
        3. Reads the matching CSV for metadata
        4. Returns full trace as a float32 tensor
    """

    def __init__(self, mat_dir: str, seq_len: int = FULL_TRACE_LEN, length: int = 100_000):
        import h5py  # noqa: F401 — validate import at init time
        self.mat_dir = Path(mat_dir)
        self.seq_len = seq_len
        self.length = length

        # discover .mat / .csv pairs (same convention as convert.py)
        mat_files = sorted(self.mat_dir.glob("*_traceMatrix.mat"))
        csv_files = sorted(self.mat_dir.glob("*_sourceMetadata.csv"))

        mat_by_id = {}
        for p in mat_files:
            plate_id = p.name.removesuffix("_traceMatrix.mat")
            mat_by_id[plate_id] = p

        csv_by_id = {}
        for p in csv_files:
            plate_id = p.name.removesuffix("_sourceMetadata.csv")
            csv_by_id[plate_id] = p

        paired_ids = sorted(set(mat_by_id) & set(csv_by_id))
        if not paired_ids:
            raise FileNotFoundError(
                f"No matched *_traceMatrix.mat / *_sourceMetadata.csv pairs in {mat_dir}"
            )

        self.pairs = [(mat_by_id[pid], csv_by_id[pid]) for pid in paired_ids]
        print(f"[RawMatDataset] {len(self.pairs)} .mat/.csv pair(s) in {mat_dir}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        import h5py
        import pandas as pd

        # pick a random pair
        pair_idx = idx % len(self.pairs)
        mat_path, csv_path = self.pairs[pair_idx]

        # open .mat, sample a random neuron
        with h5py.File(str(mat_path), "r") as f:
            n_neurons = f["normTraceMatrix"].shape[1]
            neuron_idx = np.random.randint(n_neurons)
            trace = np.asarray(
                f["normTraceMatrix"][:self.seq_len, neuron_idx], dtype=np.float32
            )

        # load matching CSV row (metadata not used by model, but we load it
        # to measure realistic IO cost)
        df = pd.read_csv(csv_path)
        _metadata = df.iloc[neuron_idx % len(df)].to_dict()  # noqa: F841

        return torch.from_numpy(trace)


# ---------------------------------------------------------------------------
# Sharded WebDataset
# ---------------------------------------------------------------------------

class ShardedWebDataset(IterableDataset):
    """IterableDataset backed by WebDataset .tar shards.

    Streams trace.npy and stimulus.npy, returns full trace.
    """

    def __init__(self, shard_dir: str, seq_len: int = FULL_TRACE_LEN):
        import webdataset as wds  # noqa: F401 — validate import at init

        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        # collect all .tar files recursively (shards may live in plate sub-dirs)
        self.shard_urls = sorted(str(p) for p in self.shard_dir.rglob("*.tar"))
        if not self.shard_urls:
            raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
        print(f"[ShardedWebDataset] {len(self.shard_urls)} shard(s) in {shard_dir}")

    def __iter__(self):
        import webdataset as wds

        seq_len = self.seq_len

        def _slice_and_tensorize(pair):
            trace, stimulus = pair
            trace = np.asarray(trace, dtype=np.float32)[:seq_len]
            return torch.from_numpy(trace)

        dataset = (
            wds.WebDataset(self.shard_urls, shardshuffle=len(self.shard_urls))
            .decode()
            .to_tuple("trace.npy", "stimulus.npy")
            .map(_slice_and_tensorize)
        )
        yield from dataset


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def _pct(part: float, total: float) -> str:
    if total == 0:
        return "N/A"
    return f"{part / total * 100:.1f}%"


def classify_bottleneck(
    t_data: float, t_transfer: float, t_forward: float,
) -> str:
    """Heuristic: the phase taking the largest share is the bottleneck."""
    phases = {"IO-bound": t_data, "CPU-bound": t_transfer, "compute-bound": t_forward}
    return max(phases, key=phases.get)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_benchmark(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    n_steps: int,
) -> None:
    """Time *n_steps* batches through load -> transfer -> forward."""

    model.eval()

    times_data = []
    times_transfer = []
    times_forward = []

    loader_iter = iter(loader)

    # warm-up: 3 batches (not timed)
    print("Warming up (3 batches) ...")
    for _ in range(3):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = batch.to(device)
        model(batch)
    torch.cuda.synchronize()

    print(f"Benchmarking {n_steps} batches ...\n")

    for step in range(n_steps):
        # -- phase 1: data loading -------------------------------------------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        t1 = time.perf_counter()
        times_data.append(t1 - t0)

        # -- phase 2: host-to-device transfer --------------------------------
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        batch = batch.to(device, non_blocking=False)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        times_transfer.append(t3 - t2)

        # -- phase 3: forward pass -------------------------------------------
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        _ = model(batch)
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        times_forward.append(t5 - t4)

    # -- aggregate stats -----------------------------------------------------
    avg_data = np.mean(times_data)
    avg_transfer = np.mean(times_transfer)
    avg_forward = np.mean(times_forward)
    avg_total = avg_data + avg_transfer + avg_forward

    print("=" * 62)
    print(f"  {'Phase':<22} {'Avg / batch':>12} {'% of total':>12}")
    print("-" * 62)
    print(f"  {'Data loading':<22} {_fmt_ms(avg_data):>12} {_pct(avg_data, avg_total):>12}")
    print(f"  {'Host -> Device':<22} {_fmt_ms(avg_transfer):>12} {_pct(avg_transfer, avg_total):>12}")
    print(f"  {'Forward pass':<22} {_fmt_ms(avg_forward):>12} {_pct(avg_forward, avg_total):>12}")
    print("-" * 62)
    print(f"  {'Total':<22} {_fmt_ms(avg_total):>12}")
    print(f"\n  Bottleneck: {classify_bottleneck(avg_data, avg_transfer, avg_forward)}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_profile(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    warmup_steps: int = 3,
    active_steps: int = 5,
) -> None:
    """Profile *active_steps* batches with torch.profiler and print results.

    Prints two tables:
        1. Top 25 CUDA kernels sorted by total GPU time
        2. Overall GPU vs CPU summary with utilization estimate
    """
    from torch.profiler import profile, ProfilerActivity, schedule

    model.eval()
    loader_iter = iter(loader)

    def get_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    # warm-up outside profiler so JIT / CUDA context init doesn't pollute
    print(f"Warming up ({warmup_steps} batches) ...")
    for _ in range(warmup_steps):
        batch = get_batch().to(device)
        model(batch)
    torch.cuda.synchronize()

    print(f"Profiling {active_steps} batches ...\n")

    prof_schedule = schedule(
        wait=0,
        warmup=1,
        active=active_steps,
        repeat=1,
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + active_steps):  # 1 warmup + active
            batch = get_batch()
            batch = batch.to(device)
            model(batch)
            torch.cuda.synchronize()
            prof.step()

    # -- print kernel-level summary ------------------------------------------
    print("=" * 80)
    print("  TOP 25 CUDA KERNELS (by total GPU time)")
    print("=" * 80)
    events = prof.key_averages()

    # PyTorch >=2.1 renamed cuda_time_total -> device_time_total
    _has_device = hasattr(events[0], "device_time_total") if events else False
    def _gpu_time(e):
        return e.device_time_total if _has_device else e.cuda_time_total

    sort_key = "device_time_total" if _has_device else "cuda_time_total"

    print(
        events.table(
            sort_by=sort_key,
            row_limit=25,
        )
    )

    # -- GPU utilization estimate --------------------------------------------
    print("\n" + "=" * 80)
    print("  GPU UTILIZATION SUMMARY")
    print("=" * 80)

    total_cuda_us = sum(_gpu_time(e) for e in events)
    total_cpu_us = sum(e.cpu_time_total for e in events)

    # wall-clock for the profiled region (active steps only)
    wall_us = total_cpu_us  # conservative upper bound
    if total_cuda_us > 0 and wall_us > 0:
        overlap_ratio = total_cuda_us / wall_us
        print(f"  Total CUDA kernel time : {total_cuda_us / 1000:.1f} ms")
        print(f"  Total CPU time         : {total_cpu_us / 1000:.1f} ms")
        print(f"  CUDA / CPU ratio       : {overlap_ratio:.2f}")
        print()
        if overlap_ratio < 0.5:
            print("  -> GPU is UNDERUTILIZED. Most time is spent on CPU / data loading.")
            print("     The GPU is idle waiting for work. Consider larger batches,")
            print("     longer sequences, or overlapping data loading with compute.")
        elif overlap_ratio < 0.9:
            print("  -> GPU utilization is MODERATE. Some CPU overhead between kernels.")
        else:
            print("  -> GPU is WELL UTILIZED. Kernel execution dominates wall time.")
    else:
        print("  No CUDA activity recorded.")

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark raw vs sharded data loading with a ~200M-param transformer.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- raw mode --
    raw_p = sub.add_parser("raw", help="Load from .mat/.csv files on the fly.")
    raw_p.add_argument(
        "--mat_dir", type=Path, required=True,
        help="Directory containing *_traceMatrix.mat and *_sourceMetadata.csv files.",
    )

    # -- sharded mode --
    shard_p = sub.add_parser("sharded", help="Stream from WebDataset .tar shards.")
    shard_p.add_argument(
        "--shard_dir", type=Path, required=True,
        help="Directory (possibly with sub-dirs) containing .tar shards.",
    )

    # -- shared args --
    for p in (raw_p, shard_p):
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--n_steps", type=int, default=50)
        p.add_argument(
            "--seq_len", type=int, default=FULL_TRACE_LEN,
            help=f"Number of timesteps to use per trace (default: {FULL_TRACE_LEN} = full trace).",
        )
        p.add_argument(
            "--profile", action="store_true",
            help="Run torch.profiler on 5 batches and print a GPU kernel summary.",
        )

    args = parser.parse_args()

    # -- device ----------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No GPU detected.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}\n")

    # -- model -----------------------------------------------------------------
    seq_len = args.seq_len
    model = TransformerModel(seq_len=seq_len).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}  ({n_params / 1e6:.1f}M)")
    print(f"Sequence length:  {seq_len}\n")

    # -- dataloader ------------------------------------------------------------
    if args.mode == "raw":
        dataset = RawMatDataset(str(args.mat_dir), seq_len=seq_len)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:  # sharded
        dataset = ShardedWebDataset(str(args.shard_dir), seq_len=seq_len)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    print(f"Mode: {args.mode}  |  batch_size={args.batch_size}  "
          f"|  num_workers={args.num_workers}  |  n_steps={args.n_steps}\n")

    # -- run -------------------------------------------------------------------
    if args.profile:
        run_profile(loader, model, device)
    else:
        run_benchmark(loader, model, device, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
