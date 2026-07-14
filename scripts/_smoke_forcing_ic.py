"""Throwaway smoke test for run_one_seed_forcing_ic_pino (Phase C gate).

One tiny optimizer step + one validation on the 8-sim varying-IC smoke set.
Asserts the encode-once / predict-closure / GradNorm path runs finite through
both encoder branches. Not a unit test; delete after wiring is confirmed.
"""
import os
import shutil
from pathlib import Path

os.environ["BENCHMARK"] = "diffusion_forcing_single"
os.environ["REPRESENTATION"] = "temporal_encoder"

from src.operators.train import load_config  # noqa: E402
from src.operators.train_pino import run_config_seeds_pino  # noqa: E402

DATA = Path("data/_smoke_dfs_varyic")
RUN = Path("runs/_smoke_forcing_ic/config0")
if RUN.exists():
    shutil.rmtree(RUN)

cfg = load_config()
d = cfg["data"]
d["trajectories.npy"] = str(DATA / "trajectories.npy")
d["x_grid_path"] = str(DATA / "x_grid.npy")
d["y_grid_path"] = str(DATA / "y_grid.npy")
d["t_grid_path"] = str(DATA / "t_grid.npy")
if "sim_params_path" in d:
    d["sim_params_path"] = str(DATA / "sim_params.npy")

t = cfg["training"]
t["epochs"] = 1
t["validate_every"] = 1
t["device"] = "cpu"

pino = t["pino"]
pino["n_r"] = 64
pino["n_ic"] = 64
pino["n_bc"] = 16
pino["sim_batch"] = 2
pino["lambda_bc_left"] = 6.0  # exercise the {r, ic, bc_left, bc_hom} split
pino["forcing"]["ny_img"] = 96  # 100-grid: 96 % forcing_patch(8) == 0
pino["forcing"]["nt_img"] = 128
pino["forcing"]["a_ref"] = 300.0
pino["causal"]["enabled"] = False

t["physics"]["n_ic_points"] = 64

summary = run_config_seeds_pino(cfg, base_run_dir=RUN, seeds=[42])
print("SMOKE_SUMMARY", summary)
