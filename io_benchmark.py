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
import urllib.request
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
# Synthetic baseline — pre-allocated GPU tensor, zero IO / zero CPU overhead
# ---------------------------------------------------------------------------

class SyntheticBatchIterator:
    """Yields the same pre-allocated GPU tensor forever.

    Unlike a DataLoader-based approach, this has:
      - Zero CPU→GPU transfer (tensor already on device)
      - Zero torch.randn overhead per step (generated once)
      - Zero DataLoader collation overhead

    This is the true ceiling: the model's compute speed with no data pipeline.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        self.batch = torch.randn(batch_size, seq_len, device=device, dtype=dtype)

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch


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


# ---------------------------------------------------------------------------
# Disk / EBS monitoring
# ---------------------------------------------------------------------------

def _find_block_device(shard_dir: str) -> str | None:
    """Find the block device backing `shard_dir` by parsing /proc/mounts."""
    shard_path = os.path.realpath(shard_dir)
    try:
        with open("/proc/mounts") as f:
            mounts = f.readlines()
    except FileNotFoundError:
        return None

    best_match = ("", "")
    for line in mounts:
        parts = line.split()
        if len(parts) < 2:
            continue
        dev, mount_point = parts[0], parts[1]
        if shard_path.startswith(mount_point) and len(mount_point) > len(best_match[1]):
            best_match = (dev, mount_point)

    dev_path = best_match[0]
    if not dev_path:
        return None

    # /dev/nvme0n1p1 → nvme0n1, /dev/xvda1 → xvda
    dev_name = os.path.basename(dev_path)
    # Strip partition suffix: nvme0n1p1 → nvme0n1, xvda1 → xvda
    if "nvme" in dev_name:
        # nvme0n1p1 → nvme0n1
        idx = dev_name.find("p", dev_name.find("n") + 1)
        if idx != -1:
            dev_name = dev_name[:idx]
    else:
        dev_name = dev_name.rstrip("0123456789")

    stat_path = f"/sys/block/{dev_name}/stat"
    if os.path.exists(stat_path):
        return dev_name
    return None


class DiskMonitor:
    """Poll /sys/block/<dev>/stat to measure read throughput in MB/s."""

    SECTOR_BYTES = 512

    def __init__(self, device_name: str | None, interval: float = 0.5):
        self.device_name = device_name
        self.interval = interval
        self.throughput_samples: list[float] = []  # MB/s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_sectors(self) -> int | None:
        if not self.device_name:
            return None
        try:
            with open(f"/sys/block/{self.device_name}/stat") as f:
                fields = f.read().split()
            # Field 2 (0-indexed) = sectors read
            return int(fields[2])
        except (FileNotFoundError, IndexError, ValueError):
            return None

    def start(self):
        self._stop.clear()
        self.throughput_samples.clear()
        if self.device_name is None:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        prev_sectors = self._read_sectors()
        prev_time = time.monotonic()
        self._stop.wait(self.interval)

        while not self._stop.is_set():
            cur_sectors = self._read_sectors()
            cur_time = time.monotonic()
            if prev_sectors is not None and cur_sectors is not None:
                dt = cur_time - prev_time
                if dt > 0:
                    delta = cur_sectors - prev_sectors
                    mb_s = (delta * self.SECTOR_BYTES) / (1024 * 1024) / dt
                    self.throughput_samples.append(mb_s)
            prev_sectors = cur_sectors
            prev_time = cur_time
            self._stop.wait(self.interval)

    def avg_read_mb_s(self) -> float:
        return float(np.mean(self.throughput_samples)) if self.throughput_samples else 0.0

    def peak_read_mb_s(self) -> float:
        return float(max(self.throughput_samples)) if self.throughput_samples else 0.0


# ---------------------------------------------------------------------------
# EC2 instance type → EBS bandwidth lookup
# ---------------------------------------------------------------------------

# Baseline EBS bandwidth in MB/s for common GPU/ML instance types.
# Source: AWS documentation for EBS-optimized instances (2025).
# For dedicated (non-burstable) instances, baseline == maximum.
# For burstable instances, the *baseline* (not burst) value is used.
# Conversion: Mbps / 8 = MB/s.
EBS_BANDWIDTH_MB_S: dict[str, float] = {
    # -- P family (GPU training) -----------------------------------------------
    "p3.2xlarge": 218.75,       # 1,750 Mbps  (dedicated)
    "p3.8xlarge": 875.0,        # 7,000 Mbps  (dedicated)
    "p3.16xlarge": 1750.0,      # 14,000 Mbps (dedicated)
    "p4d.24xlarge": 2375.0,     # 19,000 Mbps (dedicated)
    "p4de.24xlarge": 2375.0,    # 19,000 Mbps (dedicated)
    "p5.4xlarge": 1250.0,       # 10,000 Mbps (dedicated)
    "p5.48xlarge": 10000.0,     # 80,000 Mbps (dedicated)
    "p5e.48xlarge": 10000.0,    # 80,000 Mbps (dedicated)
    "p5en.48xlarge": 12500.0,   # 100,000 Mbps (dedicated)
    # -- G4ad (AMD Radeon Pro V520) --------------------------------------------
    "g4ad.xlarge": 50.0,        # 400 Mbps    (baseline, burst to 3,170)
    "g4ad.2xlarge": 100.0,      # 800 Mbps    (baseline, burst to 3,170)
    "g4ad.4xlarge": 197.5,      # 1,580 Mbps  (baseline, burst to 3,170)
    "g4ad.8xlarge": 396.25,     # 3,170 Mbps  (dedicated)
    "g4ad.16xlarge": 787.5,     # 6,300 Mbps  (dedicated)
    # -- G4dn (T4 GPU) --------------------------------------------------------
    "g4dn.xlarge": 118.75,      # 950 Mbps    (baseline, burst to 3,500)
    "g4dn.2xlarge": 143.75,     # 1,150 Mbps  (baseline, burst to 3,500)
    "g4dn.4xlarge": 593.75,     # 4,750 Mbps  (dedicated)
    "g4dn.8xlarge": 1187.5,     # 9,500 Mbps  (dedicated)
    "g4dn.12xlarge": 1187.5,    # 9,500 Mbps  (dedicated)
    "g4dn.16xlarge": 1187.5,    # 9,500 Mbps  (dedicated)
    "g4dn.metal": 2375.0,       # 19,000 Mbps (dedicated)
    # -- G5 (A10G GPU) --------------------------------------------------------
    "g5.xlarge": 87.5,          # 700 Mbps    (baseline, burst to 3,500)
    "g5.2xlarge": 106.25,       # 850 Mbps    (baseline, burst to 3,500)
    "g5.4xlarge": 593.75,       # 4,750 Mbps  (dedicated)
    "g5.8xlarge": 2000.0,       # 16,000 Mbps (dedicated)
    "g5.12xlarge": 2000.0,      # 16,000 Mbps (dedicated)
    "g5.16xlarge": 2000.0,      # 16,000 Mbps (dedicated)
    "g5.24xlarge": 2375.0,      # 19,000 Mbps (dedicated)
    "g5.48xlarge": 2375.0,      # 19,000 Mbps (dedicated)
    # -- G6 (L4 GPU) ----------------------------------------------------------
    "g6.xlarge": 125.0,         # 1,000 Mbps  (baseline, burst to 5,000)
    "g6.2xlarge": 250.0,        # 2,000 Mbps  (baseline, burst to 5,000)
    "g6.4xlarge": 1000.0,       # 8,000 Mbps  (dedicated)
    "g6.8xlarge": 2000.0,       # 16,000 Mbps (dedicated)
    "g6.12xlarge": 2500.0,      # 20,000 Mbps (dedicated)
    "g6.16xlarge": 2500.0,      # 20,000 Mbps (dedicated)
    "g6.24xlarge": 3750.0,      # 30,000 Mbps (dedicated)
    "g6.48xlarge": 7500.0,      # 60,000 Mbps (dedicated)
    # -- G6e (L40S GPU) -------------------------------------------------------
    "g6e.xlarge": 125.0,        # 1,000 Mbps  (baseline, burst to 5,000)
    "g6e.2xlarge": 250.0,       # 2,000 Mbps  (baseline, burst to 5,000)
    "g6e.4xlarge": 1000.0,      # 8,000 Mbps  (dedicated)
    "g6e.8xlarge": 2000.0,      # 16,000 Mbps (dedicated)
    "g6e.12xlarge": 2500.0,     # 20,000 Mbps (dedicated)
    "g6e.16xlarge": 2500.0,     # 20,000 Mbps (dedicated)
    "g6e.24xlarge": 3750.0,     # 30,000 Mbps (dedicated)
    "g6e.48xlarge": 7500.0,     # 60,000 Mbps (dedicated)
    # -- Gr6 (Gaudi accelerator) -----------------------------------------------
    "gr6.4xlarge": 1000.0,      # 8,000 Mbps  (dedicated)
    "gr6.8xlarge": 2000.0,      # 16,000 Mbps (dedicated)
    # -- Trn (Trainium) --------------------------------------------------------
    "trn1.2xlarge": 625.0,      # 5,000 Mbps  (baseline, burst to 20,000)
    "trn1.32xlarge": 10000.0,   # 80,000 Mbps (dedicated)
    "trn1n.32xlarge": 10000.0,  # 80,000 Mbps (dedicated)
    "trn2.48xlarge": 10000.0,   # 80,000 Mbps (dedicated)
    "trn2u.48xlarge": 10000.0,  # 80,000 Mbps (dedicated)
    # -- Inf (Inferentia) ------------------------------------------------------
    "inf2.xlarge": 156.25,      # 1,250 Mbps  (baseline, burst to 10,000)
    "inf2.8xlarge": 1250.0,     # 10,000 Mbps (dedicated)
    "inf2.24xlarge": 3750.0,    # 30,000 Mbps (dedicated)
    "inf2.48xlarge": 7500.0,    # 60,000 Mbps (dedicated)
    # -- DL (Deep Learning) ---------------------------------------------------
    "dl1.24xlarge": 2375.0,     # 19,000 Mbps (dedicated)
    "dl2q.24xlarge": 2375.0,    # 19,000 Mbps (dedicated)
}


def get_ec2_instance_type() -> str | None:
    """Query EC2 instance metadata for the instance type. Returns None outside EC2."""
    try:
        # IMDSv2: get token first
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "30"},
            method="PUT",
        )
        token = urllib.request.urlopen(req, timeout=2).read().decode()

        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=2).read().decode().strip()
    except Exception:
        return None


def get_ebs_bandwidth_limit(instance_type: str | None) -> float | None:
    """Return baseline EBS bandwidth in MB/s for the instance type, or None."""
    if instance_type is None:
        return None
    return EBS_BANDWIDTH_MB_S.get(instance_type)


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
    loader,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    n_steps: int,
    warmup_steps: int = 5,
    disk_mon: DiskMonitor | None = None,
    is_synthetic: bool = False,
) -> dict:
    """Run n_steps of forward+backward+optimizer and return timing dict.

    `loader` is either a DataLoader (real data) or a SyntheticBatchIterator.
    When is_synthetic=True the batch is already on `device`, so we skip the
    H2D transfer and measure pure compute.
    """

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

    # Warmup — ensures CUDA kernels are compiled / caches are warm
    for _ in range(warmup_steps):
        batch = next_batch()
        if not is_synthetic:
            batch = batch.to(device, non_blocking=True)
        torch.cuda.synchronize()
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed run
    gpu_mon = GpuMonitor()
    gpu_mon.start()
    if disk_mon is not None:
        disk_mon.start()

    times_data = []
    times_compute = []
    recorded_shape = None
    recorded_dtype = None

    for step in range(n_steps):
        if is_synthetic:
            # Synthetic: batch is already on GPU — no data loading cost
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = next_batch()
            t1 = time.perf_counter()
            times_data.append(t1 - t0)  # ~0, just the Python call overhead

            # Compute: forward + backward + optimizer (no H2D transfer)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            times_compute.append(t3 - t2)
        else:
            # Real data: time data fetch (CPU), H2D transfer, then compute
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = next_batch()
            # Include the H2D transfer in data time — this is the full
            # "cost of getting data onto the GPU" that we're benchmarking
            batch = batch.to(device, non_blocking=True)
            torch.cuda.synchronize()  # wait for H2D to finish
            t1 = time.perf_counter()
            times_data.append(t1 - t0)

            # Compute: forward + backward + optimizer (data already on GPU)
            t2 = time.perf_counter()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            times_compute.append(t3 - t2)

        # Record shape on first step for diagnostics
        if step == 0:
            recorded_shape = tuple(batch.shape)
            recorded_dtype = batch.dtype

    gpu_mon.stop()
    if disk_mon is not None:
        disk_mon.stop()

    avg_data = float(np.mean(times_data))
    avg_compute = float(np.mean(times_compute))
    avg_total = avg_data + avg_compute
    throughput = batch_size / avg_total if avg_total > 0 else 0.0

    result = {
        "batch_size": batch_size,
        "time_data_ms": avg_data * 1000,
        "time_compute_ms": avg_compute * 1000,
        "time_total_ms": avg_total * 1000,
        "throughput": throughput,
        "gpu_util": gpu_mon.avg_util(),
        "vram_peak_mib": gpu_mon.peak_vram_mib(),
        "vram_total_mib": gpu_mon.total_vram_mib(),
        "vram_pct": gpu_mon.peak_vram_pct(),
        "tensor_shape": recorded_shape,
        "tensor_dtype": str(recorded_dtype),
    }

    if disk_mon is not None:
        result["disk_read_avg_mb_s"] = disk_mon.avg_read_mb_s()
        result["disk_read_peak_mb_s"] = disk_mon.peak_read_mb_s()
    else:
        result["disk_read_avg_mb_s"] = 0.0
        result["disk_read_peak_mb_s"] = 0.0

    return result


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
    block_device: str | None = None,
) -> list[dict]:
    """Sweep batch sizes for one model, collecting real + synthetic results."""

    if batch_sizes is None:
        batch_sizes = auto_batch_sizes(seq_len)

    real_dataset = ShardedWebDataset(shard_dir, seq_len=seq_len)

    all_results = []
    vram_total = get_vram_total_mib()

    for bs in batch_sizes:
        print(f"\n  {model_name}  batch_size={bs}")
        print(f"  {'-' * 50}")

        # Build model fresh each batch size to get clean VRAM measurement
        model = model_cls(seq_len=seq_len).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        # --- Real data ---
        real_loader = DataLoader(
            real_dataset,
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        disk_mon = DiskMonitor(block_device)

        try:
            real = bench_config(
                real_loader, model, device, bs, n_steps,
                disk_mon=disk_mon, is_synthetic=False,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM at batch_size={bs}, stopping sweep.")
                torch.cuda.empty_cache()
                break
            raise

        real_shape = real["tensor_shape"]
        real_dtype = real["tensor_dtype"]
        disk_info = ""
        if real["disk_read_avg_mb_s"] > 0:
            disk_info = f"  Disk: {real['disk_read_avg_mb_s']:.0f} MB/s"
        print(f"    REAL   — data: {real['time_data_ms']:7.1f}ms  "
              f"compute: {real['time_compute_ms']:7.1f}ms  "
              f"throughput: {real['throughput']:8.0f} samples/s  "
              f"GPU: {real['gpu_util']:.0f}%  "
              f"VRAM: {real['vram_pct']:.0f}%{disk_info}")
        print(f"             shape: {real_shape}  dtype: {real_dtype}  "
              f"params: {n_params:,}")

        # --- Synthetic data (pre-allocated on GPU, matching real shape) ---
        # Use the actual shape from real data to guarantee identical tensors
        if real_shape is None or len(real_shape) != 2:
            print(f"    WARNING: unexpected real tensor shape {real_shape}, "
                  f"falling back to (bs, seq_len)")
            synth_shape_seq = seq_len
        else:
            synth_shape_seq = real_shape[1]

        synth_dtype = torch.float32
        if real_dtype == "torch.float64":
            synth_dtype = torch.float64
        elif real_dtype == "torch.float16":
            synth_dtype = torch.float16

        synth_iter = SyntheticBatchIterator(
            batch_size=bs, seq_len=synth_shape_seq,
            device=device, dtype=synth_dtype,
        )

        try:
            synth = bench_config(
                synth_iter, model, device, bs, n_steps,
                is_synthetic=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM on synthetic at batch_size={bs}, stopping sweep.")
                torch.cuda.empty_cache()
                break
            raise

        synth_shape = synth["tensor_shape"]
        synth_dtype_str = synth["tensor_dtype"]

        # Shape/dtype sanity check
        if real_shape != synth_shape:
            print(f"    ** SHAPE MISMATCH: real={real_shape} vs synth={synth_shape} **")
        if real_dtype != synth_dtype_str:
            print(f"    ** DTYPE MISMATCH: real={real_dtype} vs synth={synth_dtype_str} **")

        print(f"    SYNTH  — data: {synth['time_data_ms']:7.1f}ms  "
              f"compute: {synth['time_compute_ms']:7.1f}ms  "
              f"throughput: {synth['throughput']:8.0f} samples/s  "
              f"GPU: {synth['gpu_util']:.0f}%  "
              f"VRAM: {synth['vram_pct']:.0f}%")
        print(f"             shape: {synth_shape}  dtype: {synth_dtype_str}")

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

def print_summary(
    all_results: list[dict],
    num_workers: int,
    instance_type: str | None = None,
    ebs_limit_mb_s: float | None = None,
    block_device: str | None = None,
):
    """Print a final comparison table."""

    has_disk = any(r["real"].get("disk_read_avg_mb_s", 0) > 0 for r in all_results)

    print("\n")
    print("=" * 115)
    print("  SUMMARY: Real Data vs Synthetic Data Throughput")
    print("=" * 115)

    hdr = (f"  {'Model':<12} {'Batch':>6} {'Real (samp/s)':>14} {'Synth (samp/s)':>15} "
           f"{'Gap':>6} {'Data ms':>8} {'Compute ms':>11} {'GPU%':>5} {'VRAM%':>6}")
    sep = (f"  {'-'*12} {'-'*6} {'-'*14} {'-'*15} {'-'*6} {'-'*8} {'-'*11} {'-'*5} {'-'*6}")
    if has_disk:
        hdr += f"  {'Disk MB/s':>10}"
        sep += f"  {'-'*10}"
    hdr += f"  {'Status'}"
    sep += f"  {'-'*15}"
    print(hdr)
    print(sep)

    for r in all_results:
        gap = r["gap_pct"]
        if gap < 5:
            status = "OK"
        elif gap < 10:
            status = "BORDERLINE"
        else:
            status = "IO BOTTLENECK"

        line = (f"  {r['model']:<12} {r['batch_size']:>6} "
                f"{r['real']['throughput']:>14.0f} {r['synth']['throughput']:>15.0f} "
                f"{gap:>5.1f}% "
                f"{r['real']['time_data_ms']:>8.1f} {r['real']['time_compute_ms']:>11.1f} "
                f"{r['real']['gpu_util']:>5.0f} {r['real']['vram_pct']:>5.0f}%")
        if has_disk:
            line += f"  {r['real']['disk_read_avg_mb_s']:>10.0f}"
        line += f"  {status}"
        print(line)

    print("=" * 115)

    # --- EBS bandwidth analysis ---
    if instance_type or block_device:
        print(f"\n  EBS / Disk Info:")
        if instance_type:
            print(f"    Instance type: {instance_type}")
        if block_device:
            print(f"    Block device:  /dev/{block_device}")
        if ebs_limit_mb_s is not None:
            print(f"    EBS baseline bandwidth: {ebs_limit_mb_s:.0f} MB/s")
        elif instance_type:
            print(f"    EBS baseline bandwidth: unknown (instance type not in lookup table)")

    if has_disk and ebs_limit_mb_s is not None:
        peak_disk = max(r["real"].get("disk_read_peak_mb_s", 0) for r in all_results)
        ebs_usage_pct = (peak_disk / ebs_limit_mb_s) * 100 if ebs_limit_mb_s > 0 else 0

        print(f"    Peak disk read observed: {peak_disk:.0f} MB/s")
        print(f"    EBS utilization (peak):  {ebs_usage_pct:.0f}%")

        any_io_gap = any(r["gap_pct"] > 10 for r in all_results)
        if ebs_usage_pct > 80 and any_io_gap:
            print(f"\n    ** EBS BANDWIDTH BOTTLENECK **")
            print(f"    Disk reads are at {ebs_usage_pct:.0f}% of EBS baseline ({ebs_limit_mb_s:.0f} MB/s)")
            print(f"    and there is a real-vs-synthetic throughput gap.")
            print(f"    Consider: larger instance type, io2 volumes, or instance store (NVMe).")
        elif ebs_usage_pct > 80:
            print(f"\n    WARNING: Disk reads near EBS ceiling ({ebs_usage_pct:.0f}%) but no throughput gap yet.")
            print(f"    May become a bottleneck with larger models or batch sizes.")
        elif any_io_gap:
            print(f"\n    EBS bandwidth is NOT saturated ({ebs_usage_pct:.0f}% of {ebs_limit_mb_s:.0f} MB/s).")
            print(f"    IO gap is likely from CPU-side decoding or DataLoader workers, not EBS.")
    elif has_disk and ebs_limit_mb_s is None:
        peak_disk = max(r["real"].get("disk_read_peak_mb_s", 0) for r in all_results)
        print(f"\n    Peak disk read observed: {peak_disk:.0f} MB/s")
        print(f"    (Cannot determine EBS saturation — instance type not in lookup table.)")
        print(f"    Check your instance's EBS bandwidth limit in AWS docs.")

    # --- CPU / num_workers analysis ---
    worst_gap = max(all_results, key=lambda r: r["gap_pct"])

    print(f"\n  num_workers = {num_workers}")
    cpu_count = os.cpu_count() or 0
    print(f"  CPU cores available = {cpu_count}")
    if num_workers < min(4, cpu_count):
        print(f"  WARNING: num_workers={num_workers} is low relative to CPU count. "
              f"Try increasing to {min(cpu_count, 8)}.")
    if worst_gap["gap_pct"] > 10 and num_workers < cpu_count:
        ebs_not_saturated = True
        if has_disk and ebs_limit_mb_s is not None:
            peak_disk = max(r["real"].get("disk_read_peak_mb_s", 0) for r in all_results)
            ebs_not_saturated = (peak_disk / ebs_limit_mb_s) < 0.8 if ebs_limit_mb_s > 0 else True
        if ebs_not_saturated:
            print(f"  SUGGESTION: IO gap is {worst_gap['gap_pct']:.1f}% at batch_size="
                  f"{worst_gap['batch_size']}. EBS is not saturated, so try increasing "
                  f"num_workers (currently {num_workers}, system has {cpu_count} cores).")

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

    # Detect EBS / disk info
    block_device = _find_block_device(str(args.shard_dir))
    instance_type = get_ec2_instance_type()
    ebs_limit = get_ebs_bandwidth_limit(instance_type)

    print(f"Device: {gpu_name}  ({vram_total} MiB)")
    print(f"Seq length: {args.seq_len}")
    print(f"num_workers: {args.num_workers}")
    print(f"CPU cores: {os.cpu_count()}")
    if block_device:
        print(f"Block device: /dev/{block_device}")
    else:
        print(f"Block device: (not detected — disk monitoring disabled)")
    if instance_type:
        limit_str = f"{ebs_limit:.0f} MB/s" if ebs_limit else "unknown"
        print(f"EC2 instance: {instance_type}  (EBS baseline: {limit_str})")
    else:
        print(f"EC2 instance: (not detected — not on EC2 or IMDS unavailable)")
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
            block_device=block_device,
        )
        all_results.extend(results)

    print_summary(
        all_results,
        args.num_workers,
        instance_type=instance_type,
        ebs_limit_mb_s=ebs_limit,
        block_device=block_device,
    )


if __name__ == "__main__":
    main()
