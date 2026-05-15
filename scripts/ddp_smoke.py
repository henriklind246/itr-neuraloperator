"""Tiny smoke test for the distributed init/cleanup helpers.

Run single-process:
    python scripts/ddp_smoke.py

Run with torchrun:
    torchrun --standalone --nproc_per_node=2 scripts/ddp_smoke.py

Expected: each process prints its `DistInfo`. With nproc_per_node=2 you should
see rank=0 and rank=1 with world_size=2.
"""

from src.operators.distributed import cleanup_distributed, init_distributed


def main() -> int:
    dist_info = init_distributed()
    print(dist_info, flush=True)
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
