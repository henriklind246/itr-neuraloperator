"""Tests for the OOD evaluation infrastructure.

Covers the contract surface added by the OOD pipeline: per-spec OOD hooks
(`ood_axes`/`draw_latents`/`apply_ood_value`), CRN-paired generation, the two
separate time horizons, the three time pair protocols, the sidecar join in
`write_test_records`, the pooled per-simulation aggregation, sub-floor
resistance handling, and resolution accounting.
"""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import (
    OOD_LOCAL_LEAD,
    PROTOCOLS,
    SnapshotPairDataset,
    apply_protocol_pairs,
    build_protocol_pairs,
)
from data.generate_dataset import build_base_setup, run_solves
from data.generate_ood_dataset import (
    _resolution_accounting,
    classify_value,
    generate_ood_dataset,
    stable_hash,
)
from problems.registry import get_problem
from scripts.inspect_ood import BUCKET_KEYS, reduce_pairs_to_sims
from src.operators.eval import TEST_RECORD_FIELDS, write_test_records
from src.physics.internal_source import RC_MIN

# Small solver scaffolding shared by the generation-side tests. The grid must be
# wide enough for the interfaces `interface_x` extremes to clear the
# boundary-adjacent face slots, and the horizon must match the standard 0.30 so
# the pulse-train sampler has room (it draws spacings against t_final).
_TINY = dict(nx=24, ny=24, t_final=0.30, dt=0.05, save_stride=1)
_BENCHMARKS = ("forcing", "source", "source_itr", "interfaces")


def _tiny_setup(num_sims):
    return build_base_setup(
        num_sims=num_sims,
        save_stride=_TINY["save_stride"],
        nx=_TINY["nx"],
        ny=_TINY["ny"],
        t_final=_TINY["t_final"],
        dt=_TINY["dt"],
    )


def _strip(params, drop):
    """Stable hash of a params dict with the named keys removed."""
    return stable_hash({k: v for k, v in params.items() if k not in drop})


# ---------------------------------------------------------------------------
# (a) Each spec's hooks produce schema-valid, solvable sims for every sim axis.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("benchmark", _BENCHMARKS)
def test_ood_hooks_schema_valid_and_solvable(benchmark):
    spec = get_problem(benchmark, "temporal_encoder")
    axes = spec.ood_axes()
    assert axes, f"{benchmark} declares no OOD axes"

    setup = _tiny_setup(num_sims=2)
    grids = setup["grids"]
    time_cfg = setup["time_cfg"]

    sim_params = []
    for axis in axes.values():
        if axis.kind == "evaluation_parameter":
            continue  # time axis carries no sim param
        rng = np.random.default_rng(0)
        rng_profile = np.random.default_rng(1)
        latents = spec.draw_latents(rng, rng_profile, grids, time_cfg, axis)
        value = list(axis.ood_values)[0]
        params = spec.apply_ood_value(latents, axis, value)
        params.pop("_ood_realized", None)
        sim_params.append(params)

    assert sim_params
    # validate_schema must accept the OOD-injected params verbatim.
    spec.validate_schema(np.array(sim_params, dtype=object), np.arange(len(sim_params)))

    # configure_solver + a short solve must succeed for every axis value.
    trajectories, t, x, y = run_solves(
        spec, sim_params, setup["base_kwargs"], _TINY["save_stride"],
        setup["Nt_saved"], setup["Nx"], setup["Ny"], verbose=False,
    )
    assert trajectories.shape[0] == len(sim_params)
    assert np.all(np.isfinite(trajectories))


# ---------------------------------------------------------------------------
# (b) CRN invariance on ordinary axes: only the swept field changes.
# ---------------------------------------------------------------------------
# forcing, source, and interfaces all expose the identical scalar `rc` sweep.
@pytest.mark.parametrize("benchmark", ("forcing", "source", "interfaces"))
def test_crn_invariance_scalar_rc(benchmark):
    spec = get_problem(benchmark, "temporal_encoder")
    setup = _tiny_setup(num_sims=2)
    axis = spec.ood_axes()["rc"]
    assert axis.field == "R_c"
    assert axis.ood_values == (0.0, 0.01, 0.025, 1.25, 1.5, 2.0)
    latents = spec.draw_latents(
        np.random.default_rng(7), np.random.default_rng(8),
        setup["grids"], setup["time_cfg"], axis,
    )
    pa = spec.apply_ood_value(latents, axis, 1.25)
    pb = spec.apply_ood_value(latents, axis, 1.5)

    assert pa["R_c"] == 1.25 and pb["R_c"] == 1.5
    # Everything except the swept R_c (and realized geometry) is byte-identical.
    assert _strip(pa, {"R_c", "_ood_realized"}) == _strip(pb, {"R_c", "_ood_realized"})


def test_crn_invariance_temporal_axis():
    spec = get_problem("forcing", "temporal_encoder")
    setup = _tiny_setup(num_sims=2)
    axis = spec.ood_axes()["sin_freq"]
    latents = spec.draw_latents(
        np.random.default_rng(3), np.random.default_rng(4),
        setup["grids"], setup["time_cfg"], axis,
    )
    pa = spec.apply_ood_value(latents, axis, 25.0)
    pb = spec.apply_ood_value(latents, axis, 40.0)

    # The swept temporal family params differ; the background (incl. R_c) does not.
    assert pa["R_c"] == pb["R_c"]
    assert stable_hash(pa["temporal_params"]) != stable_hash(pb["temporal_params"])
    assert _strip(pa, {"temporal_params", "_ood_realized"}) == _strip(
        pb, {"temporal_params", "_ood_realized"}
    )


def test_crn_invariance_spatial_axis():
    spec = get_problem("forcing", "temporal_encoder")
    setup = _tiny_setup(num_sims=2)
    axis = spec.ood_axes()["patch_width"]
    latents = spec.draw_latents(
        np.random.default_rng(5), np.random.default_rng(6),
        setup["grids"], setup["time_cfg"], axis,
    )
    pa = spec.apply_ood_value(latents, axis, 0.05)
    pb = spec.apply_ood_value(latents, axis, 0.75)

    assert pa["R_c"] == pb["R_c"]
    assert stable_hash(pa["spatial_params"]) != stable_hash(pb["spatial_params"])
    assert _strip(pa, {"spatial_params", "_ood_realized"}) == _strip(
        pb, {"spatial_params", "_ood_realized"}
    )


# ---------------------------------------------------------------------------
# (c) CRN pairing on compound axes: field-equality cannot hold, but the
#     per-repeat shared-latent fingerprint matches across values.
# ---------------------------------------------------------------------------
def _latents_hash_by_repeat(save_path):
    by_repeat = {}
    with open(save_path / "ood_metadata.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            by_repeat.setdefault(rec["ood_repeat"], set()).add(rec["latents_hash"])
    return by_repeat


@pytest.mark.parametrize(
    "benchmark,axis,values",
    [
        ("interfaces", "family_transfer", [("sin", "uniform"), ("exp", "patch")]),
        ("source_itr", "rc_severity", [0.1, 1.5]),
    ],
)
def test_crn_pairing_compound_axis(tmp_path, benchmark, axis, values):
    save_path = generate_ood_dataset(
        benchmark=benchmark,
        axis_name=axis,
        save_dir=tmp_path / f"{benchmark}_{axis}",
        values=values,
        repeats=2,
        dataset_t_final=_TINY["t_final"],
        dt=_TINY["dt"],
        save_stride=_TINY["save_stride"],
        nx=_TINY["nx"],
        ny=_TINY["ny"],
        rng_seed=0,
        verbose=False,
    )
    by_repeat = _latents_hash_by_repeat(save_path)
    assert len(by_repeat) == 2
    # Within a repeat every swept value shares one shared-latent hash.
    for hashes in by_repeat.values():
        assert len(hashes) == 1
    # Different repeats draw different backgrounds.
    all_hashes = {next(iter(h)) for h in by_repeat.values()}
    assert len(all_hashes) == 2


# ---------------------------------------------------------------------------
# Shared dataset for the separate-horizons / protocol tests. A 0.05-spaced grid
# lands 0.0/0.25/0.30/0.35/0.40/0.45 exactly on snapshots.
# ---------------------------------------------------------------------------
@pytest.fixture
def horizon_dataset(synthetic_sim_params):
    n_t = 10
    x_grid = np.linspace(0.0, 1.0, 11)
    y_grid = np.linspace(0.0, 1.0, 11)
    t_grid = np.linspace(0.0, 0.45, n_t)  # spacing 0.05
    trajectories = np.zeros((2, n_t, 11, 11), dtype=np.float32)
    spec = get_problem("forcing", "temporal_encoder")
    return SnapshotPairDataset(
        trajectories,
        t_grid,
        x_grid,
        y_grid,
        np.arange(2),
        synthetic_sim_params[:2],
        0.0,
        1.0,
        n_snapshots=None,
        dt=0.05,
        t_final=0.45,
        time_norm_horizon=0.30,
        problem=spec,
    )


def _find_pair(dataset, **match):
    for idx, tag in enumerate(dataset._pair_tags):
        if all(np.isclose(tag[k], v) if isinstance(v, float) else tag[k] == v
               for k, v in match.items()):
            return idx
    raise AssertionError(f"no pair matching {match}")


# ---------------------------------------------------------------------------
# (d) Separate horizons: ds.t_final stays physical (0.45) while temporal
#     features divide by time_norm_horizon (0.30). Assert exact closed forms.
# ---------------------------------------------------------------------------
def test_separate_horizons_closed_form(horizon_dataset):
    ds = horizon_dataset
    assert ds.t_final == pytest.approx(0.45)
    assert ds.time_norm_horizon == pytest.approx(0.30)

    tags = build_protocol_pairs(
        ds, list(PROTOCOLS), [0.25, 0.30, 0.35, 0.40, 0.45],
        horizon=0.30, atol=0.0125,
    )
    apply_protocol_pairs(ds, tags)

    def cond(**match):
        return ds[_find_pair(ds, **match)]["cond_static"].numpy()

    # fixed_initial, t_s=0: lead = t_t/0.30.
    c = cond(protocol="fixed_initial", target_time_actual=0.30)
    assert c[0] == pytest.approx(1.0, abs=2e-3)  # lead in-range at ID target
    c = cond(protocol="fixed_initial", target_time_actual=0.45)
    assert c[0] == pytest.approx(1.5, abs=2e-3)  # lead > 1 (OOD)

    # anchored_from_horizon, t_s=0.30: lead=(0.45-0.30)/0.30=0.5.
    c = cond(protocol="anchored_from_horizon", target_time_actual=0.45)
    assert c[0] == pytest.approx(0.5, abs=2e-3)

    # ood_local_fixed_lead, t_s=0.35, t_t=0.40: lead=0.05/0.30.
    c = cond(protocol="ood_local_fixed_lead", source_time_actual=0.35)
    assert c[0] == pytest.approx(0.05 / 0.30, abs=2e-3)


# ---------------------------------------------------------------------------
# (e) Protocol structure: each protocol pins a single source definition; the
#     aggregation key separates source/lead so curves never pool tasks.
# ---------------------------------------------------------------------------
def test_protocol_structure(horizon_dataset):
    ds = horizon_dataset
    tags = build_protocol_pairs(
        ds, list(PROTOCOLS), [0.25, 0.30, 0.35, 0.40, 0.45],
        horizon=0.30, atol=0.0125,
    )
    by_proto = {p: [t for t in tags if t["protocol"] == p] for p in PROTOCOLS}

    # fixed_initial: only t_s=0, spans both ID and OOD targets (5 targets x 2 sims).
    fi = by_proto["fixed_initial"]
    assert len(fi) == 10
    assert all(t["source_time_actual"] == pytest.approx(0.0) for t in fi)
    fi_targets = {round(t["target_time_actual"], 2) for t in fi}
    assert {0.25, 0.30} <= fi_targets  # ID-reference targets present

    # anchored_from_horizon: only t_s=0.30, OOD targets only (3 x 2 sims).
    an = by_proto["anchored_from_horizon"]
    assert len(an) == 6
    assert all(t["source_time_actual"] == pytest.approx(0.30) for t in an)
    assert all(t["target_time_actual"] > 0.30 for t in an)

    # ood_local_fixed_lead: source past horizon, fixed lead (2 sources x 2 sims).
    ol = by_proto["ood_local_fixed_lead"]
    assert len(ol) == 4
    assert all(t["source_time_actual"] > 0.30 for t in ol)
    assert all(t["lead_time_actual"] == pytest.approx(OOD_LOCAL_LEAD, abs=2e-3) for t in ol)

    # Source and lead are part of the aggregation key, so tasks never pool.
    assert "source_time_actual" in BUCKET_KEYS
    assert "lead_time_actual" in BUCKET_KEYS


# ---------------------------------------------------------------------------
# (f) Sidecar join: write_test_records reads OOD identity from
#     ood_metadata.jsonl by sim_id and leaves sim_params untouched.
# ---------------------------------------------------------------------------
def test_write_test_records_sidecar_join(
    tmp_path, small_fno2d, synthetic_trajectories, synthetic_sim_params
):
    trajectories, x_grid, y_grid, t_grid = synthetic_trajectories
    np.save(tmp_path / "trajectories.npy", trajectories)
    np.save(tmp_path / "x_grid.npy", x_grid)
    np.save(tmp_path / "y_grid.npy", y_grid)
    np.save(tmp_path / "t_grid.npy", t_grid)
    np.save(tmp_path / "sim_params.npy", synthetic_sim_params)

    # OOD sidecar covering every sim id; absent entries would default in-distribution.
    with open(tmp_path / "ood_metadata.jsonl", "w") as f:
        for sim_id in range(trajectories.shape[0]):
            f.write(json.dumps({
                "sim_id": sim_id,
                "ood_axis": "rc",
                "ood_value": 1.5,
                "ood_repeat": 0,
                "latents_hash": "deadbeef",
                "distribution_class": "out_of_distribution",
            }) + "\n")

    conf = {
        "data": {
            "trajectories.npy": str(tmp_path / "trajectories.npy"),
            "x_grid_path": str(tmp_path / "x_grid.npy"),
            "y_grid_path": str(tmp_path / "y_grid.npy"),
            "t_grid_path": str(tmp_path / "t_grid.npy"),
            "sim_params_path": str(tmp_path / "sim_params.npy"),
        },
        "training": {
            "batch_size": 4,
            "n_snapshots_test": 3,
            "device": "cpu",
            "loss": {"interface_half_width": 0.05},
        },
        "model": {
            "parameters": {
                "modes1": 2, "modes2": 2, "width": 8, "in_channels": 4,
                "out_channels": 1, "n_layers": 2, "cond_static_dim": 10,
                "cond_hidden": 256, "temporal_token_dim": 2,
                "temporal_samples": 128, "temporal_hidden": 16,
                "forcing_embed_dim": 16,
            }
        },
    }
    seed_dir = tmp_path / "seed42"
    seed_dir.mkdir()
    torch.save(
        {
            "model_state": small_fno2d.state_dict(),
            "conf": conf,
            "best_val": 1.23,
            "mu_global": 0.0,
            "sigma_global": 1.0,
        },
        seed_dir / "fno2d_best.pt",
    )

    out_path = write_test_records(tmp_path)
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == TEST_RECORD_FIELDS
        rows = list(reader)
    assert rows
    # Every emitted row inherits the sidecar OOD identity by sim_id.
    assert all(r["ood_axis"] == "rc" for r in rows)
    assert all(r["distribution_class"] == "out_of_distribution" for r in rows)

    # sim_params on disk is unchanged (no OOD pollution).
    reloaded = np.load(tmp_path / "sim_params.npy", allow_pickle=True)
    assert "ood_axis" not in reloaded[0]


# ---------------------------------------------------------------------------
# (g) Per-simulation aggregation pools squared error; this differs from the
#     secondary mean-of-pair-RMSEs on a multi-pair sim.
# ---------------------------------------------------------------------------
def test_pooled_rmse_differs_from_mean_pair_rmse():
    key = {k: 0 for k in BUCKET_KEYS}
    key.update(
        seed=42, benchmark="forcing", ood_axis="rc", ood_value=1.5,
        ood_repeat=0, distribution_class="out_of_distribution",
        protocol="fixed_initial", source_time_actual=0.0,
        target_time_actual=0.30, lead_time_actual=0.30, sim_id=3,
    )
    base = dict(
        key,
        target_sse_K2=100.0, interface_sse_K2=0.0, num_interface_cells=0,
        interface_target_sse_K2=0.0, eff_cells_per_scale=np.nan,
        eff_steps_per_scale=np.nan, time_norm_horizon=0.30, dataset_t_final=0.45,
    )
    df = pd.DataFrame([
        {**base, "sse_K2": 4.0, "num_error_cells": 1, "rmse_K": 2.0},
        {**base, "sse_K2": 16.0, "num_error_cells": 1, "rmse_K": 4.0},
    ])
    out = reduce_pairs_to_sims(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_pairs"] == 2
    # Pooled: sqrt((4+16)/(1+1)) = sqrt(10); mean-of-pairs: (2+4)/2 = 3.
    assert row["rmse_K"] == pytest.approx(np.sqrt(10.0))
    assert row["mean_pair_rmse_K"] == pytest.approx(3.0)
    assert row["rmse_K"] != pytest.approx(row["mean_pair_rmse_K"])


# ---------------------------------------------------------------------------
# (h) Zero / sub-floor resistance. The log-normalized source_itr rc_base axis
#     rejects values below RC_MIN; the linearly-normalized scalar-R_c axes
#     accept R_c=0 as a labeled limiting case.
# ---------------------------------------------------------------------------
def test_source_itr_rc_base_rejects_subfloor():
    spec = get_problem("source_itr", "temporal_encoder")
    setup = _tiny_setup(num_sims=2)
    axis = spec.ood_axes()["rc_base"]
    latents = spec.draw_latents(
        np.random.default_rng(0), np.random.default_rng(1),
        setup["grids"], setup["time_cfg"], axis,
    )
    with pytest.raises(ValueError):
        spec.apply_ood_value(latents, axis, 0.5 * RC_MIN)


def test_scalar_rc_accepts_zero_and_solves():
    spec = get_problem("forcing", "temporal_encoder")
    setup = _tiny_setup(num_sims=1)
    axis = spec.ood_axes()["rc"]
    latents = spec.draw_latents(
        np.random.default_rng(2), np.random.default_rng(3),
        setup["grids"], setup["time_cfg"], axis,
    )
    params = spec.apply_ood_value(latents, axis, 0.0)
    params.pop("_ood_realized", None)
    assert params["R_c"] == 0.0
    # Perfect contact is a labeled limiting case, not a boundary/in-dist value.
    assert classify_value(axis, 0.0) == "limiting_case"

    spec.validate_schema(np.array([params], dtype=object), np.arange(1))
    traj, *_ = run_solves(
        spec, [params], setup["base_kwargs"], _TINY["save_stride"],
        setup["Nt_saved"], setup["Nx"], setup["Ny"], verbose=False,
    )
    assert np.all(np.isfinite(traj))


# ---------------------------------------------------------------------------
# (i) Resolution accounting routes the swept feature to the right per-feature
#     criterion (cells vs steps) and leaves non-resolution axes empty.
# ---------------------------------------------------------------------------
def test_resolution_accounting_per_feature_type():
    spec = get_problem("forcing", "temporal_encoder")
    setup = _tiny_setup(num_sims=2)
    grids, time_cfg = setup["grids"], setup["time_cfg"]
    dy = float(grids["y_grid"][1] - grids["y_grid"][0])
    dt = float(setup["dt"])
    axes = spec.ood_axes()

    def acct(name, value):
        axis = axes[name]
        latents = spec.draw_latents(
            np.random.default_rng(0), np.random.default_rng(1), grids, time_cfg, axis,
        )
        params = spec.apply_ood_value(latents, axis, value)
        params.pop("_ood_realized", None)
        return _resolution_accounting(spec, axis, params, dy, dt)

    # Temporal axis -> steps populated, cells empty.
    a = acct("sin_freq", 40.0)
    assert a["eff_steps_per_scale"] is not None
    assert a["eff_cells_per_scale"] is None
    assert a["resolution_criterion"] == "steps_per_period"

    # Spatial axis -> cells populated, steps empty.
    a = acct("patch_width", 0.05)
    assert a["eff_cells_per_scale"] is not None
    assert a["eff_steps_per_scale"] is None
    assert a["resolution_criterion"] == "cells_per_patch_width"

    # Non-resolution axis -> all empty.
    a = acct("rc", 1.5)
    assert a["eff_cells_per_scale"] is None
    assert a["eff_steps_per_scale"] is None
    assert a["resolution_criterion"] is None
