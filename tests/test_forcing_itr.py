import copy
import math

import numpy as np
import pytest
import torch

from data.dataset import SnapshotPairDataset
from problems.registry import get_problem
from scripts import invert
from scripts.inverse_adapters import ForcingItrAdapter, InverseAdapter
from src.operators.fno2d import FNO2d
from src.physics.internal_source import (
    RC_MIN,
    RC_SIN_RANGES,
    R_PEAK_MAX,
    equivalent_scalar_resistance,
    integrated_excess_resistance,
    interface_control_volume_weights,
    make_rc_sin_profile,
    make_rc_void_profile,
)


def _sample_pair(num_sims=6, nx=12, ny=12, t_final=0.3, dt=0.005):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grids = {"X": X, "Y": Y, "x_grid": x, "y_grid": y}
    time_cfg = {
        "num_sims": num_sims,
        "dt": dt,
        "t_final": t_final,
        "lhs_seed": 0,
        "forcing_profile_seed": 1,
        "t_on": 0.0,
        "t_off": 0.2,
        "phase": 0.0,
        "tukey_alpha": 0.5,
    }
    forcing = get_problem("forcing")
    forcing_itr = get_problem("forcing_itr")
    parent = forcing.sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1), grids, time_cfg
    )
    spatial = forcing_itr.sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1), grids, time_cfg
    )
    return forcing, forcing_itr, parent, spatial, grids, time_cfg


def _dataset(spec, params, grids, representation="temporal_encoder"):
    n = len(params)
    t_grid = np.array([0.0, 0.07, 0.15, 0.30], dtype=np.float32)
    trajectories = np.zeros(
        (n, len(t_grid), grids["X"].shape[0], grids["X"].shape[1]),
        dtype=np.float32,
    )
    return SnapshotPairDataset(
        trajectories=trajectories,
        sim_params=np.asarray(params, dtype=object),
        t_grid=t_grid,
        x_grid=grids["x_grid"],
        y_grid=grids["y_grid"],
        sim_ids=np.arange(n),
        mu_global=0.0,
        sigma_global=1.0,
        problem=get_problem(spec.name, representation),
        dt=0.005,
        ramp_seconds=0.01,
    )


def _base_kwargs(grids, *, t_final=0.02, dt=0.005):
    nx, ny = grids["X"].shape
    return {
        "a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0,
        "Nx": nx, "Ny": ny, "lam_target": 0.8,
        "t_final": t_final, "flux_f": 0.0, "flux_A": 0.0,
        "t_on": 0.0, "t_off": 0.2, "phase": 0.0,
        "dt": dt, "tukey_alpha": 0.5,
        "y_grid": grids["y_grid"], "X": grids["X"], "Y": grids["Y"],
        "ramp_seconds": 0.01,
    }


def test_forcing_itr_pairing_provenance_and_forcing_params():
    _, spec, parent, spatial, _, _ = _sample_pair()
    for index, (base, profile) in enumerate(zip(parent, spatial)):
        assert profile["parent_benchmark"] == "forcing"
        assert profile["parent_sim_id"] == index
        assert profile["forcing_sample_key"] == (
            f"forcing-v1:seed=1:index={index:06d}"
        )
        assert profile["R_c_base"] == pytest.approx(base["R_c"])
        assert profile["temporal_family"] == base["temporal_family"]
        assert profile["temporal_params"] == base["temporal_params"]
        assert profile["spatial_family"] == base["spatial_family"]
        assert profile["spatial_params"] == base["spatial_params"]
    spec.validate_schema(np.asarray(spatial, dtype=object), np.arange(len(spatial)))


def test_forcing_itr_inherits_exact_forcing_tensors():
    representation = "temporal_encoder"
    forcing, forcing_itr, parent, spatial, grids, _ = _sample_pair()
    parent_ds = _dataset(forcing, parent, grids, representation)
    spatial_ds = _dataset(forcing_itr, spatial, grids, representation)
    parent_item = parent_ds.problem.build_item(parent_ds, 0, 0, 3)
    spatial_item = spatial_ds.problem.build_item(spatial_ds, 0, 0, 3)

    np.testing.assert_array_equal(
        spatial_item["forcing_seq"], parent_item["forcing_seq"]
    )
    np.testing.assert_array_equal(
        spatial_item["spatial"][..., 3], parent_item["spatial"][..., 3]
    )
    np.testing.assert_array_equal(
        spatial_item["spatial"][..., :4], parent_item["spatial"][..., :4]
    )
    np.testing.assert_array_equal(
        spatial_item["spatial"][..., 5:], parent_item["spatial"][..., 4:]
    )
    np.testing.assert_array_equal(
        spatial_item["cond_static"][5:], parent_item["cond_static"][2:]
    )
    assert spatial_item["forcing_seq"].shape == (
        128, parent_ds.problem.dims.temporal_token_dim
    )


def test_forcing_itr_alias_is_inert_and_canonical_fields_are_required():
    _, spec, _, spatial, grids, _ = _sample_pair()
    ds = _dataset(spec, spatial, grids)
    before = ds.problem.build_item(ds, 0, 0, 3)
    ds.sim_params[0]["R_c"] = 99.0
    after = ds.problem.build_item(ds, 0, 0, 3)
    np.testing.assert_array_equal(after["cond_static"], before["cond_static"])
    np.testing.assert_array_equal(after["spatial"], before["spatial"])

    bad = copy.deepcopy(ds.sim_params)
    del bad[0]["R_c_base"]
    with pytest.raises(ValueError, match="missing keys"):
        spec.validate_schema(bad, np.array([0]))


def test_forcing_itr_solver_uses_flat_profile_and_zero_amp_matches_scalar():
    forcing, forcing_itr, parent, spatial, grids, _ = _sample_pair(
        num_sims=1, t_final=0.02
    )
    spatial[0]["R_c_amp"] = 0.0
    kwargs = _base_kwargs(grids)
    scalar_solver = forcing.configure_solver(parent[0], kwargs)
    profile_solver = forcing_itr.configure_solver(spatial[0], kwargs)
    assert len(profile_solver.interface_R) == 1
    assert np.asarray(profile_solver.interface_R[0]).shape == (
        len(grids["y_grid"]),
    )
    scalar = scalar_solver.solve(T0=parent[0]["T0"], store_trajectory=True)[-1]
    profile = profile_solver.solve(T0=spatial[0]["T0"], store_trajectory=True)[-1]
    np.testing.assert_allclose(profile, scalar, rtol=1e-11, atol=1e-11)


def test_forcing_itr_zero_amp_encoding_is_finite():
    _, spec, _, spatial, grids, _ = _sample_pair(num_sims=1)
    spatial[0]["R_c_amp"] = 0.0
    ds = _dataset(spec, spatial, grids)
    item = ds.problem.build_item(ds, 0, 0, 3)
    assert np.all(np.isfinite(item["cond_static"]))
    assert np.all(np.isfinite(item["spatial"][..., 4]))
    assert item["cond_static"][2] == pytest.approx(0.0)


def _sample_pair_sin(num_sims=6, nx=12, ny=12, t_final=0.3, dt=0.005):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grids = {"X": X, "Y": Y, "x_grid": x, "y_grid": y}
    time_cfg = {
        "num_sims": num_sims, "dt": dt, "t_final": t_final,
        "lhs_seed": 0, "forcing_profile_seed": 1,
        "t_on": 0.0, "t_off": 0.2, "phase": 0.0, "tukey_alpha": 0.5,
    }
    forcing = get_problem("forcing")
    sin = get_problem("forcing_itr_sin")
    parent = forcing.sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1), grids, time_cfg
    )
    spatial = sin.sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1), grids, time_cfg
    )
    return forcing, sin, parent, spatial, grids, time_cfg


def test_forcing_itr_sin_provenance_schema_and_bounds():
    _, spec, parent, spatial, _, _ = _sample_pair_sin()
    for index, (base, profile) in enumerate(zip(parent, spatial)):
        assert profile["parent_benchmark"] == "forcing"
        assert profile["parent_sim_id"] == index
        assert profile["R_c_base"] == pytest.approx(base["R_c"])
        assert "R_c_A" in profile
        for key in ("R_c_amp", "R_c_y0", "R_c_sigma"):
            assert key not in profile
        R_base = float(profile["R_c_base"])
        A = float(profile["R_c_A"])
        assert 0.0 <= A <= R_PEAK_MAX - R_base + 1e-9
    spec.validate_schema(np.asarray(spatial, dtype=object), np.arange(len(spatial)))


def test_forcing_itr_sin_inherits_forcing_tensors_and_cond_layout():
    representation = "temporal_encoder"
    forcing, sin, parent, spatial, grids, _ = _sample_pair_sin()
    parent_ds = _dataset(forcing, parent, grids, representation)
    sin_ds = _dataset(sin, spatial, grids, representation)
    parent_item = parent_ds.problem.build_item(parent_ds, 0, 0, 3)
    sin_item = sin_ds.problem.build_item(sin_ds, 0, 0, 3)
    # R_c(y) channel inserted at index 4; forcing channels otherwise identical.
    np.testing.assert_array_equal(
        sin_item["spatial"][..., :4], parent_item["spatial"][..., :4]
    )
    np.testing.assert_array_equal(
        sin_item["spatial"][..., 5:], parent_item["spatial"][..., 4:]
    )
    # cond [t_bar, R_base, A, profile_bins(8)]; the block after the two
    # sinusoid scalars matches the parent's block after its scalar R_c slot.
    np.testing.assert_array_equal(
        sin_item["cond_static"][3:], parent_item["cond_static"][2:]
    )
    assert sin_item["cond_static"].shape == (11,)
    assert sin_item["spatial"].shape[-1] == 5


def test_forcing_itr_sin_schema_rejects_void_keys():
    _, spec, _, spatial, _, _ = _sample_pair_sin(num_sims=2)
    bad = copy.deepcopy(np.asarray(spatial, dtype=object))
    bad[0]["R_c_amp"] = 0.5
    with pytest.raises(ValueError, match="forbidden keys"):
        spec.validate_schema(bad, np.array([0]))


def test_forcing_itr_sin_solver_uses_sin_profile():
    _, sin, _, spatial, grids, _ = _sample_pair_sin(num_sims=1, t_final=0.02)
    kwargs = _base_kwargs(grids)
    solver = sin.configure_solver(spatial[0], kwargs)
    assert len(solver.interface_R) == 1
    prof = np.asarray(solver.interface_R[0])
    assert prof.shape == (len(grids["y_grid"]),)
    expected = make_rc_sin_profile(
        grids["y_grid"],
        R_base=float(spatial[0]["R_c_base"]),
        A=float(spatial[0]["R_c_A"]),
    )
    np.testing.assert_allclose(prof, expected, rtol=1e-12, atol=1e-12)


def test_fv_weights_severity_continuous_limit_and_general_domain_req():
    y = np.linspace(0.0, 1.0, 101)
    weights = interface_control_volume_weights(y, (0.0, 1.0))
    h = y[1] - y[0]
    np.testing.assert_allclose(
        weights, np.r_[h / 2.0, np.full(y.size - 2, h), h / 2.0]
    )
    R_base, R_amp, y0, sigma = 0.3, 1.2, 0.2, 0.1
    profile = make_rc_void_profile(y, R_base, R_amp, y0, sigma)
    discrete = integrated_excess_resistance(
        y, profile, R_base, bounds=(0.0, 1.0)
    )
    analytic = (
        0.5 * math.sqrt(math.pi) * R_amp * sigma
        * (
            math.erf((1.0 - y0) / sigma)
            + math.erf(y0 / sigma)
        )
    )
    assert discrete == pytest.approx(analytic, rel=2e-3)

    centers = np.array([-0.5, 0.5, 1.5])
    center_weights = interface_control_volume_weights(centers, (-1.0, 2.0))
    np.testing.assert_allclose(center_weights, 1.0)
    assert equivalent_scalar_resistance(
        centers, np.full(3, 2.0), bounds=(-1.0, 2.0)
    ) == pytest.approx(2.0)


def test_rc_sin_profile_invariants_peak_and_severity():
    base_lo, base_hi = RC_SIN_RANGES["R_base"]
    rng = np.random.default_rng(0)
    y = np.linspace(0.0, 1.0, 100)  # deliberately excludes y = 0.5
    for _ in range(16):
        R_base = float(rng.uniform(base_lo, base_hi))
        A = float(rng.uniform(0.0, 1.0) * (R_PEAK_MAX - R_base))
        # Sampling invariants.
        assert RC_MIN <= R_base <= R_PEAK_MAX
        assert 0.0 <= A <= R_PEAK_MAX - R_base + 1e-12

        profile = make_rc_sin_profile(y, R_base, A)
        assert profile.min() >= RC_MIN - 1e-12
        assert profile.max() <= R_PEAK_MAX + 1e-12
        # A general grid need not contain y = 0.5, so the array max only bounds
        # the true peak; assert the exact peak by evaluating at the point.
        assert profile.max() <= R_base + A + 1e-12
        peak = make_rc_sin_profile(np.array([0.5]), R_base, A)[0]
        np.testing.assert_allclose(peak, R_base + A)

        # Numeric FV severity vs analytic int_0^1 (R_c - R_base) dy = A * 2 / pi.
        y_fine = np.linspace(0.0, 1.0, 401)
        prof_fine = make_rc_sin_profile(y_fine, R_base, A)
        computed = integrated_excess_resistance(
            y_fine, prof_fine, R_base, bounds=(0.0, 1.0)
        )
        np.testing.assert_allclose(computed, A * 2.0 / np.pi, rtol=2e-3, atol=1e-6)


def test_interface_weights_match_solver_dy():
    _, spec, _, spatial, grids, _ = _sample_pair(num_sims=1)
    solver = spec.configure_solver(spatial[0], _base_kwargs(grids))
    np.testing.assert_allclose(
        interface_control_volume_weights(grids["y_grid"], (0.0, 1.0)),
        solver.dy,
    )


def test_physical_time_and_interface_pair_resolution_cross_grid():
    t_grid = np.array([0.0, 0.07, 0.15, 0.30])
    indices, resolved = invert.resolve_observation_times(
        t_grid, [0.07, 0.15, 0.30]
    )
    assert indices == [1, 2, 3]
    np.testing.assert_array_equal(resolved, [0.07, 0.15, 0.30])
    with pytest.raises(ValueError, match="not uniquely present"):
        invert.resolve_observation_times(t_grid, [0.08])

    t_grid_float32 = np.array([0.0, 0.07, 0.15, 0.30], dtype=np.float32)
    indices, resolved_float32 = invert.resolve_observation_times(
        t_grid_float32, [0.07, 0.15, 0.30]
    )
    assert indices == [1, 2, 3]
    indices, _ = invert.resolve_observation_times(
        t_grid, resolved_float32
    )
    assert indices == [1, 2, 3]

    requested_y = np.linspace(0.1, 0.9, 8)
    mask_a, meta_a = invert.build_interface_pair_mask(
        np.linspace(0.0, 1.0, 100),
        np.linspace(0.0, 1.0, 100),
        requested_y=requested_y,
    )
    mask_b, meta_b = invert.build_interface_pair_mask(
        np.linspace(0.0, 1.0, 160),
        np.linspace(0.0, 1.0, 160),
        requested_y=requested_y,
    )
    assert int(mask_a.sum()) == int(mask_b.sum()) == 16
    assert np.all(np.abs(meta_a["resolved_y"] - requested_y) <= 0.5 / 99)
    assert np.all(np.abs(meta_b["resolved_y"] - requested_y) <= 0.5 / 159)
    assert meta_a["resolved_x"][0] < 0.5 < meta_a["resolved_x"][1]
    assert meta_b["resolved_x"][0] < 0.5 < meta_b["resolved_x"][1]


def test_forcing_itr_inverse_dispatch_and_determinism():
    adapter = InverseAdapter.from_config(
        {"benchmark": {"name": "forcing_itr"}}
    )
    assert isinstance(adapter, ForcingItrAdapter)
    torch.manual_seed(4)
    model = FNO2d(
        modes1=2, modes2=2, width=8, in_channels=5, out_channels=1,
        n_layers=2, cond_static_dim=13, temporal_token_dim=3,
        temporal_hidden=16, forcing_embed_dim=16,
        use_forcing_time_aug=True, s_y_channel=3,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spatial = torch.zeros(3, 12, 12, 5)
    spatial[..., 3] = 1.0
    cond = torch.zeros(3, 13)
    cond[:, 0] = torch.tensor([0.07, 0.15, 0.30])
    forcing_seq = torch.zeros(3, 128, 3)
    forcing_seq[..., 0] = torch.linspace(0.0, 1.0, 128)
    obs = invert.ObservationSet(
        sid=0, time_indices=[1, 2, 3], spatial=spatial, cond=cond,
        forcing_seq=forcing_seq, targets=torch.zeros(3, 12, 12, 1),
        y_grid=torch.linspace(0.0, 1.0, 12), Nx=12,
    )
    outputs = []
    gradients = []
    for _ in range(2):
        theta = torch.tensor(
            [0.3, 1.0, 0.5, 0.12], requires_grad=True
        )
        prediction = invert.predict_fullfield(model, obs, theta, adapter)
        loss = prediction.square().mean()
        gradient = torch.autograd.grad(loss, theta)[0]
        outputs.append(prediction.detach())
        gradients.append(gradient.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0.0, atol=0.0)


def test_calibration_artifact_rejects_mismatch_and_uses_frozen_scale(tmp_path):
    artifact = {
        "calibration_schema_version": np.int64(2),
        "benchmark": np.str_("forcing_itr"),
        "split": np.str_("val"),
        "sigma_fno_norm": np.float64(0.02),
        "noise_std_norm": np.float64(0.05),
        "checkpoint_fingerprint": np.str_("checkpoint"),
        "dataset_fingerprint": np.str_("dataset"),
        "observation_fingerprint": np.str_("sensors"),
        "local_gate_pass": np.bool_(True),
    }
    path = tmp_path / "calibration.npz"
    invert.write_surrogate_calibration(str(path), artifact)
    loaded = invert.load_surrogate_calibration(
        str(path), benchmark="forcing_itr",
        checkpoint_fingerprint="checkpoint",
        dataset_fingerprint="dataset",
        observation_fingerprint="sensors",
        noise_std_norm=0.05,
    )
    sigma = float(loaded["sigma_fno_norm"])
    assert invert.effective_sigma2(
        0.05, c_fno=999.0, sigma_fno_cal=sigma
    ) == pytest.approx(0.05**2 + 0.02**2)
    with pytest.raises(ValueError, match="dataset_fingerprint mismatch"):
        invert.load_surrogate_calibration(
            str(path), benchmark="forcing_itr",
            checkpoint_fingerprint="checkpoint",
            dataset_fingerprint="other",
            observation_fingerprint="sensors",
            noise_std_norm=0.05,
        )


def test_validation_calibration_artifact_construction():
    _, spec, _, spatial, grids, _ = _sample_pair(num_sims=3)
    ds = _dataset(spec, spatial, grids)
    ds._split_ids = {
        "train": np.array([0]),
        "val": np.array([1, 2]),
        "test": np.array([], dtype=int),
    }
    adapter = ForcingItrAdapter()
    torch.manual_seed(5)
    model = FNO2d(
        modes1=2, modes2=2, width=8, in_channels=5, out_channels=1,
        n_layers=2, cond_static_dim=13, temporal_token_dim=3,
        temporal_hidden=16, forcing_embed_dim=16,
        use_forcing_time_aug=True, s_y_channel=3,
    ).eval()
    calibration = invert.build_surrogate_calibration(
        model, ds, adapter,
        time_indices=[1, 2, 3],
        observation_times=np.array([0.07, 0.15, 0.30]),
        sensor_layout="interface_pair",
        sensor_y=np.linspace(0.1, 0.9, 8),
        interface_x=0.5,
        sensor_x_halfwidth=None,
        sensor_n_y=8,
        mu_global=0.0,
        sigma_global=2.0,
        noise_std_norm=0.05,
        checkpoint_fingerprint="checkpoint",
        dataset_fingerprint="dataset",
        device="cpu",
    )
    assert calibration["split"] == "val"
    assert int(calibration["simulation_count"]) == 2
    assert int(calibration["sample_count"]) == 2 * 3 * 16
    assert np.asarray(calibration["theta_true"]).shape == (2, 4)
    assert np.asarray(calibration["S_R"]).shape == (2,)
    assert np.asarray(calibration["interface_jump_rms_K"]).shape == (2,)
    assert float(calibration["sigma_fno_K"]) == pytest.approx(
        2.0 * float(calibration["sigma_fno_norm"])
    )
    # The correlation-aware design effect is frozen into the artifact and is
    # never below the independent baseline of 1.
    assert int(calibration["residual_dim"]) == 3 * 16
    assert float(calibration["residual_design_effect"]) >= 1.0
    assert float(calibration["residual_n_eff"]) == pytest.approx(
        float(calibration["residual_dim"])
        / float(calibration["residual_design_effect"])
    )
