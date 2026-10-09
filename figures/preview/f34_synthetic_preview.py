"""SYNTHETIC layout preview for F34. Not model output.

Writes fake schema-v2 test_records.csv for the four benchmarks, renders F34
through the real visual.pub pipeline (manifest -> stats -> figure -> sidecar),
then saves a watermarked copy to figures/preview/.

Run from the repo root:
  .venv/bin/python figures/preview/f34_synthetic_preview.py
"""

import argparse
import tempfile
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from visual.pub import records, registry, style  # noqa: E402
from visual.pub.manifest import FigureRequirements, Manifest  # noqa: E402

KEY = "F34_global_field_error_lead_panels"
N_SIMS = 150
GRID = np.round(np.arange(31) * 0.01, 10)  # t in [0, 0.3], as in the paper
CELLS = 100 * 100


def median_curve(benchmark, lead):
    """Invented median RMSE [K] against lead, scaled loosely to Table 4."""
    if benchmark == "forcing":
        return 0.012 + 0.050 * (1 - np.exp(-lead / 0.08))
    if benchmark == "interfaces":
        # Short-lead maximum from the varying initial condition, then growth.
        return 0.045 * np.exp(-lead / 0.015) + 0.020 + 0.035 * (lead / 0.3) ** 1.2
    if benchmark == "source":
        heating = 0.006 * np.exp(-((lead - 0.12) / 0.05) ** 2)
        return 0.004 + 0.016 * (1 - np.exp(-lead / 0.12)) + heating
    return 0.007 + 0.028 * (1 - np.exp(-lead / 0.10))


SIM_SIGMA = {"forcing": 0.55, "interfaces": 0.50, "source": 0.60, "source_itr_sin": 0.65}


def synthetic_records(benchmark, rng):
    s_idx, j_idx = np.triu_indices(len(GRID), k=1)
    t_s, t_j = GRID[s_idx], GRID[j_idx]
    lead = t_j - t_s
    # Later source snapshots start from a developed field and are easier.
    source_factor = 1.0 / (1.0 + 2.0 * t_s / 0.3)
    frames = []
    for sim in range(N_SIMS):
        sim_factor = np.exp(rng.normal(0.0, SIM_SIGMA[benchmark]))
        # Each case also gets its own growth rate, so the spread widens with lead.
        slope = np.exp(rng.normal(0.0, 0.25))
        noise = np.exp(rng.normal(0.0, 0.08, size=lead.size))
        rmse = (median_curve(benchmark, lead) * sim_factor
                * (1 + (slope - 1) * lead / 0.3) * source_factor * noise)
        frame = pd.DataFrame({name: np.nan for name in records.SCHEMA_V1_COLUMNS},
                             index=range(lead.size))
        if benchmark in ("forcing", "interfaces"):
            r_c = rng.uniform(0.05, 1.0)
        else:
            r_c = rng.uniform(17.5, 350.0)
        frame = frame.assign(
            seed="0", benchmark=benchmark, sim_id=sim, s=s_idx, j=j_idx,
            t_s=t_s, t_bar=lead, R_c=r_c,
            R_c_A=rng.uniform(0, 300.0) if benchmark == "source_itr_sin" else np.nan,
            rmse_K=rmse, sse_K2=rmse ** 2 * CELLS, num_error_cells=CELLS,
            target_sse_K2=(5.0 * (1 + 20 * t_j)) ** 2 * CELLS,
            interface_sse_K2=rmse ** 2, num_interface_cells=1,
            interface_target_sse_K2=100.0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/preview")
    parser.add_argument("--work", default=tempfile.mkdtemp(prefix="f34_synthetic_"))
    args = parser.parse_args()

    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    rng = np.random.default_rng(20261009)
    sources = {}
    for benchmark in ("forcing", "source", "source_itr_sin", "interfaces"):
        run = work / "runs" / f"synthetic_{benchmark}" / "config0" / "seed0"
        run.mkdir(parents=True)
        synthetic_records(benchmark, rng).to_csv(run / "test_records.csv", index=False)
        (run / "config_used.yaml").write_text(yaml.safe_dump({
            "benchmark": {"name": benchmark, "representation": "temporal_encoder"},
            "evaluation": {"rollout": {"enabled": False}}}))
        sources[f"{benchmark}_records"] = [{"run": str(run), "seed": 0}]
    manifest = Manifest(sources=sources, requirements=FigureRequirements.load())

    # Full pipeline: provenance sidecar and statistics CSV land in the work dir.
    result = registry.render(KEY, manifest=manifest, out_dir=work / "pub_out",
                             strict=False, force=True)
    print("pipeline render:", *[str(p) for p in result.paths], result.sidecar, sep="\n  ")

    # Watermarked preview copy for the repo.
    spec = registry.get_figure(KEY)
    source = manifest.resolve(KEY, strict=False)
    with style.pub_style():
        fig, _, definition = spec.load()(source=source, spec=spec,
                                         requirement=manifest.requirements.figures[KEY],
                                         **spec.params)
    fig.text(0.5, 0.5, "SYNTHETIC DATA", transform=fig.transFigure, ha="center",
             va="center", fontsize=22, color="0.5", alpha=0.12, rotation=40,
             fontweight="bold", zorder=0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with style.pub_style(extra={"savefig.bbox": None}):
        for ext in ("png", "pdf"):
            path = out / f"F34_SYNTHETIC_preview.{ext}"
            fig.savefig(path)
            print("preview:", path)
    for b, g in definition["growth"].items():
        print(f"{b:15s} n={g['n_simulations']:3d} {g['first_median_rmse_K']:.4f} -> "
              f"{g['last_median_rmse_K']:.4f} K  x{g['ratio_last_to_first']:.2f}  "
              f"peak {g['peak_median_rmse_K']:.4f} at {g['peak_lead']:.2f}")


if __name__ == "__main__":
    main()
