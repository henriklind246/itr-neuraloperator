"""Distributed (DDP) initialization helpers.

Detection is environment-based: torchrun sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
`MASTER_ADDR`, `MASTER_PORT`. When `WORLD_SIZE > 1` the process group is
initialized; otherwise the helpers return a single-process `DistInfo` so the
existing single-GPU code paths are unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

# Default process-group collective timeout. NCCL's library default (10 min) is
# too tight: rank-0-only validation (full val set on one GPU + per-pair CSV
# writes) can leave other ranks idling at the post-validation broadcast for
# longer than that, tripping the watchdog and aborting the job. Override per-run
# with the DDP_TIMEOUT_MIN environment variable (set in the DDP sbatch).
_DEFAULT_DDP_TIMEOUT_MIN = 30.0


def _resolve_timeout() -> timedelta:
    raw = os.environ.get("DDP_TIMEOUT_MIN")
    minutes = _DEFAULT_DDP_TIMEOUT_MIN
    if raw is not None:
        try:
            minutes = float(raw)
        except ValueError:
            minutes = _DEFAULT_DDP_TIMEOUT_MIN
    return timedelta(minutes=minutes)


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool


_DIST_INFO: DistInfo | None = None


def init_distributed() -> DistInfo:
    """Initialize the process group if torchrun env vars are set; else single-process.

    Idempotent: subsequent calls return the cached `DistInfo`.
    """
    global _DIST_INFO
    if _DIST_INFO is not None:
        return _DIST_INFO

    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if world_size_env is not None and int(world_size_env) > 1:
        world_size = int(world_size_env)
        rank = int(rank_env) if rank_env is not None else 0
        local_rank = int(local_rank_env) if local_rank_env is not None else 0

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=_resolve_timeout(),
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        _DIST_INFO = DistInfo(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_distributed=True,
        )
    else:
        _DIST_INFO = DistInfo(rank=0, world_size=1, local_rank=0, is_distributed=False)

    return _DIST_INFO


def get_dist_info() -> DistInfo:
    """Return the cached `DistInfo`, initializing single-process if needed."""
    if _DIST_INFO is None:
        return init_distributed()
    return _DIST_INFO


def cleanup_distributed() -> None:
    """Tear down the process group. Safe to call in single-process mode."""
    global _DIST_INFO
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    _DIST_INFO = None


def is_main_process() -> bool:
    return get_dist_info().rank == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
