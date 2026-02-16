"""GPU IO bottleneck benchmark for WebDataset data loading pipeline.

Tests whether our shard-based data loading can keep up with the fastest
plausible model architectures by comparing real data throughput against
synthetic (torch.randn) throughput.  The gap between them is the IO cost.

Models tested (intentionally fast / cheap to maximise IO pressure):
    1. ShallowCNN  — 3-layer 1D conv, tiny
    2. SimpleMLP   — 2-layer fully-connected

For each model the script sweeps batch sizes upward until VRAM hits ~85-90%,
then prints a summary table showing real vs synthetic throughput and the
percentage gap at every batch size.

Usage
-----
    python io_benchmark.py --shard_dir /data/shards
    python io_benchmark.py --shard_dir /data/shards --seq_len 512 --num_workers 8
    python io_benchmark.py --shard_dir /data/shards --batch_sizes 32 64 128 256
"""

import argparse
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

FULL_TRACE_LEN = 11481


# ---------------------------------------------------------------------------
# Models — deliberately fast so IO is the bottleneck if anything is
# ---------------------------------------------------------------------------

class ShallowCNN(nn.Module):
    """3-layer 1D CNN.  Very fast forward/backward → maximum IO pressure."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len)
        x = x.unsqueeze(1)       # (B, 1, seq_len)
        x = self.net(x)          # (B, 1, seq_len)
        return x.squeeze(1)      # (B, seq_len)


class SimpleMLP(nn.Module):
    """2-layer MLP.  Even cheaper than the CNN."""

    def __init__(self, seq_len: int):
        super().__init__()
        hidden = min(seq_len, 1024)
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# WebDataset loader  (mirrors the shard format from convert.py)
# ---------------------------------------------------------------------------

class ShardedWebDataset(IterableDataset):
    """Stream trace tensors from .tar shards written by convert.py."""

    def __init__(self, shard_dir: str, seq_len: int = FULL_TRACE_LEN):
        import webdataset as wds  # noqa: F401

        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.shard_urls = sorted(str(p) for p in self.shard_dir.rglob("*.tar"))
        if not self.shard_urls:
            raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
        print(f"[ShardedWebDataset] {len(self.shard_urls)} shard(s) in {shard_dir}")

    def __iter__(self):
        import webdataset as wds

        seq_len = self.seq_len

        def _to_tensor(pair):
            trace, _stimulus = pair
            trace = np.asarray(trace, dtype=np.float32)[:seq_len]
            return torch.from_numpy(trace)

        dataset = (
            wds.WebDataset(self.shard_urls, shardshuffle=len(self.shard_urls))
            .decode()
            .to_tuple("trace.npy", "stimulus.npy")
            .map(_to_tensor)
        )
        yield from dataset


# ---------------------------------------------------------------------------
# Synthetic (torch.randn) loader — baseline with zero IO
# ---------------------------------------------------------------------------

class SyntheticDataset(IterableDataset):
    """Infinite stream of random tensors — no IO at all."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            yield torch.randn(self.seq_len)


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

class GpuMonitor:
    """Poll GPU utilization + VRAM usage via nvidia-smi in a background thread."""

    def __init__(self, device_index: int = 0, interval: float = 0.25):
        self.device_index = device_index
        self.interval = interval
        self.util_samples: list[int] = []
        self.vram_samples: list[tuple[int, int]] = []  # (used_MiB, total_MiB)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self.util_samples.clear()
        self.vram_samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                )
                parts = [x.strip() for x in out.strip().split(",")]
                self.util_samples.append(int(parts[0]))
                self.vram_samples.append((int(parts[1]), int(parts[2])))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def avg_util(self) -> float:
        return float(np.mean(self.util_samples)) if self.util_samples else 0.0

    def peak_vram_pct(self) -> float:
        if not self.vram_samples:
            return 0.0
        peak_used = max(u for u, _ in self.vram_samples)
        total = self.vram_samples[0][1]
        return peak_used / total * 100 if total > 0 else 0.0

    def peak_vram_mib(self) -> int:
        if not self.vram_samples:
            return 0
        return max(u for u, _ in self.vram_samples)

    def total_vram_mib(self) -> int:
        if not self.vram_samples:
            return 0
        return self.vram_samples[0][1]


def get_vram_used_mib() -> int:
    """Snapshot of current VRAM usage."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() // (1024 * 1024)


def get_vram_total_mib() -> int:
    return torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)


# ---------------------------------------------------------------------------
# Single-configuration benchmark
# ---------------------------------------------------------------------------

def bench_config(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    n_steps: int,
    warmup_steps: int = 5,
) -> dict:
    """Run n_steps of forward+backward+optimizer and return timing dict."""

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    loader_iter = iter(loader)

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    # Warmup
    for _ in range(warmup_steps):
        batch = next_batch().to(device)
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed run
    gpu_mon = GpuMonitor()
    gpu_mon.start()

    times_data = []
    times_compute = []

    for _ in range(n_steps):
        # --- data loading time ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = next_batch()
        t1 = time.perf_counter()
        times_data.append(t1 - t0)

        # --- compute time (transfer + forward + backward + optim) ---
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        times_compute.append(t3 - t2)

    gpu_mon.stop()

    avg_data = float(np.mean(times_data))
    avg_compute = float(np.mean(times_compute))
    avg_total = avg_data + avg_compute
    throughput = batch_size / avg_total if avg_total > 0 else 0.0

    return {
        "batch_size": batch_size,
        "time_data_ms": avg_data * 1000,
        "time_compute_ms": avg_compute * 1000,
        "time_total_ms": avg_total * 1000,
        "throughput": throughput,
        "gpu_util": gpu_mon.avg_util(),
        "vram_peak_mib": gpu_mon.peak_vram_mib(),
        "vram_total_mib": gpu_mon.total_vram_mib(),
        "vram_pct": gpu_mon.peak_vram_pct(),
    }


# ---------------------------------------------------------------------------
# Batch-size sweep
# ---------------------------------------------------------------------------

VRAM_TARGET_PCT = 88  # stop sweeping when VRAM exceeds this


def auto_batch_sizes(seq_len: int) -> list[int]:
    """Generate batch sizes to sweep: powers of 2 from 16 up to 4096."""
    sizes = []
    bs = 16
    while bs <= 4096:
        sizes.append(bs)
        bs *= 2
    return sizes


def sweep_model(
    model_cls,
    model_name: str,
    shard_dir: str,
    seq_len: int,
    num_workers: int,
    n_steps: int,
    batch_sizes: list[int] | None,
    device: torch.device,
) -> list[dict]:
    """Sweep batch sizes for one model, collecting real + synthetic results."""

    if batch_sizes is None:
        batch_sizes = auto_batch_sizes(seq_len)

    real_dataset = ShardedWebDataset(shard_dir, seq_len=seq_len)
    synth_dataset = SyntheticDataset(seq_len)

    all_results = []
    vram_total = get_vram_total_mib()

    for bs in batch_sizes:
        print(f"\n  {model_name}  batch_size={bs}")
        print(f"  {'-' * 50}")

        # Build model fresh each batch size to get clean VRAM measurement
        model = model_cls(seq_len=seq_len).to(device)

        # --- Real data ---
        real_loader = DataLoader(
            real_dataset,
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        try:
            real = bench_config(real_loader, model, device, bs, n_steps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM at batch_size={bs}, stopping sweep.")
                torch.cuda.empty_cache()
                break
            raise

        print(f"    REAL   — data: {real['time_data_ms']:7.1f}ms  "
              f"compute: {real['time_compute_ms']:7.1f}ms  "
              f"throughput: {real['throughput']:8.0f} samples/s  "
              f"GPU: {real['gpu_util']:.0f}%  "
              f"VRAM: {real['vram_pct']:.0f}%")

        # --- Synthetic data ---
        synth_loader = DataLoader(
            synth_dataset,
            batch_size=bs,
            num_workers=0,  # no IO needed
            pin_memory=True,
            drop_last=True,
        )

        try:
            synth = bench_config(synth_loader, model, device, bs, n_steps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM on synthetic at batch_size={bs}, stopping sweep.")
                torch.cuda.empty_cache()
                break
            raise

        print(f"    SYNTH  — data: {synth['time_data_ms']:7.1f}ms  "
              f"compute: {synth['time_compute_ms']:7.1f}ms  "
              f"throughput: {synth['throughput']:8.0f} samples/s  "
              f"GPU: {synth['gpu_util']:.0f}%  "
              f"VRAM: {synth['vram_pct']:.0f}%")

        gap = (1 - real["throughput"] / synth["throughput"]) * 100 if synth["throughput"] > 0 else 0
        print(f"    GAP    — {gap:.1f}%  {'<< IO bottleneck!' if gap > 10 else 'OK' if gap < 5 else 'borderline'}")

        all_results.append({
            "model": model_name,
            "batch_size": bs,
            "real": real,
            "synth": synth,
            "gap_pct": gap,
        })

        del model
        torch.cuda.empty_cache()

        # Stop if VRAM is past the target
        if real["vram_pct"] > VRAM_TARGET_PCT:
            print(f"    VRAM at {real['vram_pct']:.0f}% — reached target, stopping sweep.")
            break

    return all_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict], num_workers: int):
    """Print a final comparison table."""

    print("\n")
    print("=" * 100)
    print("  SUMMARY: Real Data vs Synthetic Data Throughput")
    print("=" * 100)
    print(f"  {'Model':<12} {'Batch':>6} {'Real (samp/s)':>14} {'Synth (samp/s)':>15} "
          f"{'Gap':>6} {'Data ms':>8} {'Compute ms':>11} {'GPU%':>5} {'VRAM%':>6}  {'Status'}")
    print(f"  {'-'*12} {'-'*6} {'-'*14} {'-'*15} {'-'*6} {'-'*8} {'-'*11} {'-'*5} {'-'*6}  {'-'*15}")

    for r in all_results:
        gap = r["gap_pct"]
        if gap < 5:
            status = "OK"
        elif gap < 10:
            status = "BORDERLINE"
        else:
            status = "IO BOTTLENECK"

        print(f"  {r['model']:<12} {r['batch_size']:>6} "
              f"{r['real']['throughput']:>14.0f} {r['synth']['throughput']:>15.0f} "
              f"{gap:>5.1f}% "
              f"{r['real']['time_data_ms']:>8.1f} {r['real']['time_compute_ms']:>11.1f} "
              f"{r['real']['gpu_util']:>5.0f} {r['real']['vram_pct']:>5.0f}%  "
              f"{status}")

    print("=" * 100)

    # Check for CPU / num_workers bottleneck
    worst_data = max(all_results, key=lambda r: r["real"]["time_data_ms"])
    worst_gap = max(all_results, key=lambda r: r["gap_pct"])

    print(f"\n  num_workers = {num_workers}")
    cpu_count = os.cpu_count() or 0
    print(f"  CPU cores available = {cpu_count}")
    if num_workers < min(4, cpu_count):
        print(f"  WARNING: num_workers={num_workers} is low relative to CPU count. "
              f"Try increasing to {min(cpu_count, 8)}.")
    if worst_gap["gap_pct"] > 10 and num_workers < cpu_count:
        print(f"  SUGGESTION: IO gap is {worst_gap['gap_pct']:.1f}% at batch_size="
              f"{worst_gap['batch_size']}. Try increasing num_workers (currently {num_workers}, "
              f"system has {cpu_count} cores).")

    any_bad = any(r["gap_pct"] > 10 for r in all_results)
    any_borderline = any(5 <= r["gap_pct"] <= 10 for r in all_results)
    print()
    if not any_bad and not any_borderline:
        print("  VERDICT: IO is NOT a bottleneck. Data loading keeps up at all tested batch sizes.")
    elif any_bad:
        print("  VERDICT: IO IS a bottleneck at some batch sizes. See rows marked IO BOTTLENECK.")
    else:
        print("  VERDICT: IO is borderline. Probably fine but worth monitoring under real training.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU IO bottleneck benchmark — real WebDataset shards vs synthetic data.",
    )
    parser.add_argument(
        "--shard_dir", type=Path, required=True,
        help="Directory (with sub-dirs) containing .tar shards from convert.py.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=FULL_TRACE_LEN,
        help=f"Trace length to use per sample (default: {FULL_TRACE_LEN}).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader num_workers (default: 4).",
    )
    parser.add_argument(
        "--n_steps", type=int, default=30,
        help="Timed training steps per configuration (default: 30).",
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=None,
        help="Explicit batch sizes to test (default: auto-sweep 16..4096).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    vram_total = get_vram_total_mib()

    print(f"Device: {gpu_name}  ({vram_total} MiB)")
    print(f"Seq length: {args.seq_len}")
    print(f"num_workers: {args.num_workers}")
    print(f"CPU cores: {os.cpu_count()}")
    print()

    all_results = []

    for model_cls, model_name in [(ShallowCNN, "ShallowCNN"), (SimpleMLP, "SimpleMLP")]:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        n_params = sum(p.numel() for p in model_cls(seq_len=args.seq_len).parameters())
        print(f"  Parameters: {n_params:,}")

        results = sweep_model(
            model_cls=model_cls,
            model_name=model_name,
            shard_dir=str(args.shard_dir),
            seq_len=args.seq_len,
            num_workers=args.num_workers,
            n_steps=args.n_steps,
            batch_sizes=args.batch_sizes,
            device=device,
        )
        all_results.extend(results)

    print_summary(all_results, args.num_workers)


if __name__ == "__main__":
    main()
