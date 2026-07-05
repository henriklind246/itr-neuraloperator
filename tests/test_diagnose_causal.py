import csv

import numpy as np
import torch

from problems.diffusion import K_SLAB
from scripts.diagnose_causal import (
    REGION_KEYS,
    accumulate_bin_residuals,
    build_temporal_bin_spec,
    ic_anchor_error,
    solution_error_by_lead,
    write_bin_csv,
    write_solution_csv,
)

"""Smoke coverage for the Step 0.5 baseline diagnostic (scripts/diagnose_causal.py).

Exercises the importable core (bin-spec construction, per-bin residual
accumulation, IC-anchor error, solution-error-by-lead, and the CSV writers) with
a stub sampler + identity model so the diagnostic runs and emits the expected
columns without a real checkpoint. No numerical assertion on residual values.
"""


class _IdentityModel(torch.nn.Module):
    def forward(self, spatial, cond_static, forcing_seq=None):
        return spatial[..., 0:1]


class _BinnedStubColloc:
    """Minimal on-grid-style sampler that records ``_last_leads`` and
    ``_last_lead_bin_ids`` from an attached ``bin_spec`` exactly like the real
    CollocationSampler, so ``accumulate_bin_residuals`` can bin its draws."""

    def __init__(self, B=8, Nx=5, Ny=5, dt=0.01, n_t=11, seed=0):
        self.B, self.Nx, self.Ny = B, Nx, Ny
        xg = np.linspace(0.0, 1.0, Nx)
        yg = np.linspace(0.0, 1.0, Ny)
        self.dt = float(dt)
        self.t_final = float((n_t - 1) * dt)
        self.n_max = n_t - 2
        self.geom_cfg = {
            "x_grid": xg, "y_grid": yg,
            "k_left": K_SLAB, "k_right": K_SLAB,
            "interface_x": float(xg[Nx // 2]),
            "dt": self.dt, "sigma_global": 1.0, "T_right_tilde": 0.0,
        }
        self.base_plan = {"base_snapshot_index": 0, "on_grid_pairs": True}
        self.bin_spec = None
        self._last_leads = None
        self._last_lead_bin_ids = None
        self.rng = np.random.default_rng(seed)

    def _mk_batch(self, field):
        spatial = np.zeros((self.B, self.Nx, self.Ny, 4), np.float32)
        spatial[..., 0] = field
        return {
            "spatial": torch.from_numpy(spatial),
            "cond_static": torch.zeros(self.B, 2),
            "forcing_seq": torch.zeros(self.B, 128, 2),
        }

    def sample_batch(self, max_lead):
        n_max = max(0, min(int(np.floor(float(max_lead) / self.dt)) - 1, self.n_max))
        ns = self.rng.integers(0, n_max + 1, size=self.B)
        self._last_leads = (ns + 0.5) * self.dt
        self._last_lead_bin_ids = (
            self.bin_spec.interval_to_bin(self._last_leads)
            if self.bin_spec is not None else None
        )
        f = (self.rng.standard_normal((self.B, self.Nx, self.Ny)) * 0.01).astype(np.float32)
        z = torch.zeros(self.B, self.Ny)
        return (self._mk_batch(f), self._mk_batch(f), torch.zeros(self.B),
                z, z.clone(), z.clone())

    def sample_anchor_batch(self):
        f = self.rng.standard_normal((self.B, self.Nx, self.Ny)).astype(np.float32)
        batch = self._mk_batch(f)
        target = torch.from_numpy(f[..., None].astype(np.float32))
        return batch, target


def test_build_temporal_bin_spec_on_grid_domain():
    sampler = _BinnedStubColloc(dt=0.01, n_t=11)
    spec = build_temporal_bin_spec(sampler, n_bins=6)
    assert spec.n_bins == 6
    assert abs(spec.lead_lo - 0.5 * 0.01) < 1e-12
    assert abs(spec.lead_hi - (0.1 - 0.5 * 0.01)) < 1e-12


def test_accumulate_bin_residuals_shapes_and_counts():
    sampler = _BinnedStubColloc(B=8, n_t=11, seed=1)
    spec = build_temporal_bin_spec(sampler, n_bins=5)
    means, counts = accumulate_bin_residuals(
        _IdentityModel(), sampler, spec, torch.device("cpu"),
        n_batches=4, max_lead=sampler.t_final,
    )
    assert set(means.keys()) == set(REGION_KEYS)
    for k in REGION_KEYS:
        assert means[k].shape == (5,)
        assert np.all(np.isfinite(means[k]))
    assert counts.shape == (5,)
    # Every drawn row lands in exactly one bin.
    assert int(counts.sum()) == 4 * 8


def test_ic_anchor_error_zero_for_identity():
    sampler = _BinnedStubColloc(seed=2)
    err = ic_anchor_error(_IdentityModel(), sampler, torch.device("cpu"))
    # Identity model reproduces the base field exactly -> zero IC error.
    assert err == 0.0


def test_solution_error_by_lead_columns():
    class _DS:
        def __init__(self):
            self.t_grid = np.linspace(0.0, 0.1, 11)
            self.trajectories = np.full((3, 11, 5, 5), 300.0)

    ds = _DS()
    # predict_fn returns a normalized field; with mu=300, sigma=1 the Kelvin
    # field is ~300, matching the equilibrium trajectory (dev -> 0 -> NaN pert).
    def _predict(sid, t_hi):
        return np.zeros((5, 5))

    rows = solution_error_by_lead(_predict, ds, [0, 1, 2], [1, 5, 10],
                                  mu_global=300.0, sigma_global=1.0)
    assert len(rows) == 3
    for r in rows:
        assert set(r.keys()) == {"lead", "rmse_K", "dev_from_eq_K", "pert_rel_err", "n_sims"}
        assert r["n_sims"] == 3


def test_csv_writers_emit_expected_columns(tmp_path):
    sampler = _BinnedStubColloc(seed=3)
    spec = build_temporal_bin_spec(sampler, n_bins=4)
    means, counts = accumulate_bin_residuals(
        _IdentityModel(), sampler, spec, torch.device("cpu"),
        n_batches=2, max_lead=sampler.t_final,
    )
    bin_path = tmp_path / "bin_residuals.csv"
    write_bin_csv(bin_path, spec, means, counts)
    with bin_path.open() as f:
        header = next(csv.reader(f))
    assert header == (
        ["bin_id", "lead_lo", "lead_hi", "lead_mid", "count"]
        + [f"{k}_resid" for k in REGION_KEYS]
    )

    rows = [{"lead": 0.05, "rmse_K": 1.0, "dev_from_eq_K": 2.0,
             "pert_rel_err": 0.5, "n_sims": 4}]
    sol_path = tmp_path / "solution_error.csv"
    write_solution_csv(sol_path, rows)
    with sol_path.open() as f:
        header = next(csv.reader(f))
    assert header == ["lead", "rmse_K", "dev_from_eq_K", "pert_rel_err", "n_sims"]
