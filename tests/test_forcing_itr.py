import copy

import numpy as np
import pytest
import torch

from data.dataset import SnapshotPairDataset
from problems.registry import get_problem
from scripts import invert
from scripts.inverse_adapters import ForcingItrSinAdapter, InverseAdapter
from src.operators.fno2d import FNO2d
from src.physics.internal_source import (
    RC_MIN,
    RC_SIN_RANGES,
    R_PEAK_MAX,
    equivalent_scalar_resistance,
    interface_control_volume_weights,
    make_rc_sin_profile,
)




def _dataset(spec, params, grids, representation="temporal_encoder", time_scale=1.0):
    n = len(params)
    t_grid = np.array([0.0, 0.07, 0.15, 0.30], dtype=np.float32) * time_scale
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
        ramp_seconds=0.01 * time_scale,
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


def test_b5_stretches_every_forcing_family_and_preserves_conditioning():
    from src.physics.boundary_forcing import build_qL, build_qL_integral

    _, short, _, reference, grids, time_cfg = _sample_pair_sin(num_sims=64)
    long = get_problem("forcing_itr_sin_long")
    stretched = long.sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1), grids,
        dict(time_cfg, t_final=7.0),
    )
    assert {p["temporal_family"] for p in reference} == {
        "sin", "exp", "pulse_train", "exp_train",
    }
    short_ds = _dataset(short, reference, grids)
    long_ds = _dataset(long, stretched, grids, time_scale=7.0 / 0.3)
    times = np.linspace(0.0, 0.3, 97)
    for sid, (old, new) in enumerate(zip(reference, stretched)):
        for key in ("R_c", "R_c_base", "R_c_A", "spatial_family", "spatial_params"):
            assert old[key] == new[key]
        np.testing.assert_array_equal(old["T0"], new["T0"])
        assert new["forcing_time_scale"] == 7.0 / 0.3
        fields = ("temporal_family", "temporal_params", "spatial_family", "spatial_params")
        old_kwargs = {key: old[key] for key in fields}
        new_kwargs = {key: new[key] for key in fields}
        old_flux, _ = build_qL(**old_kwargs, y_grid=grids["y_grid"], t_ramp=0.01)
        new_flux, _ = build_qL(**new_kwargs, y_grid=grids["y_grid"], t_ramp=0.01 * 7.0 / 0.3)
        np.testing.assert_allclose(
            [new_flux((7.0 / 0.3) * t) for t in times],
            [old_flux(t) for t in times], rtol=1e-10, atol=1e-10,
        )
        old_integral, _ = build_qL_integral(**old_kwargs, y_grid=grids["y_grid"], t_ramp=0.01)
        new_integral, _ = build_qL_integral(**new_kwargs, y_grid=grids["y_grid"], t_ramp=0.01 * 7.0 / 0.3)
        np.testing.assert_allclose(new_integral(0.0, 7.0), (7.0 / 0.3) * old_integral(0.0, 0.3))
        old_item = short.build_item(short_ds, sid, 0, 3)
        new_item = long.build_item(long_ds, sid, 0, 3)
        for key in ("spatial", "cond_static", "forcing_seq"):
            np.testing.assert_allclose(new_item[key], old_item[key], rtol=1e-4, atol=1e-5)


def test_b5_generation_and_inverse_use_1400_steps_and_31_snapshots(tmp_path, monkeypatch):
    from data.generate_dataset import main as generate_main
    from data.dataset import assert_dataset_problem_version
    from src.operators.train import load_config

    monkeypatch.setenv("BENCHMARK", "forcing_itr_sin_long")
    config = load_config()
    spec = get_problem(config["benchmark"]["name"])
    assert config["data"]["t_final"] == spec.default_t_final == 7.0
    assert config["data"]["save_stride"] is None
    assert config["data"]["saved_snapshots"] == spec.default_saved_snapshots == 31
    assert config["data"]["ramp_seconds"] == pytest.approx(spec.default_ramp_seconds)
    assert spec.default_ramp_seconds == pytest.approx(7.0 / 30.0)
    generate_main([
        "--num-sims", "1", "--nx", "12", "--ny", "12",
        "--save-dir", str(tmp_path),
    ])
    ds = invert.build_dataset_from_dir(
        str(tmp_path), config, mu_global=300.0, sigma_global=1.0,
        n_invert=1, n_calibration=0,
    )
    assert ds.trajectories.shape == (1, 31, 12, 12)
    saved_steps = np.rint(np.linspace(0, 1400, 31)).astype(int)
    np.testing.assert_allclose(ds.t_grid, saved_steps * 0.005)
    assert set(np.diff(saved_steps)) == {46, 47}
    np.testing.assert_allclose(ds.t_grid, np.linspace(0.0, 7.0, 31), atol=0.0025)
    assert ds.dt == 0.005
    assert ds.ramp_seconds == pytest.approx(7.0 / 30.0)
    indices, _ = invert.resolve_observation_times(ds.t_grid, spec.required_observation_times)
    assert indices == [7, 15, 30]
    adapter = InverseAdapter.from_config(config)
    adapter.validate_dataset(ds)
    params = ds.sim_params[0]
    theta = adapter.theta_from_sim_params(params, dtype=torch.float64)
    solver = adapter.build_fv_solver(ds, params, theta)
    assert len(solver.t) - 1 == 1400
    assert solver.dt == 0.005
    _, _, _, expected = solver.solve(T0=params["T0"], store_trajectory=True)
    np.testing.assert_array_equal(ds.trajectories[0], expected[saved_steps].astype(np.float32))
    meta = assert_dataset_problem_version(spec, tmp_path / "t_grid.npy")
    assert meta["problem_version"] == "forcing-itr-sin-long-7s-v1"
    (tmp_path / "meta.npy").unlink()
    with pytest.raises(ValueError, match="regenerate"):
        assert_dataset_problem_version(spec, tmp_path / "t_grid.npy")


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


def test_fv_weights_and_general_domain_equivalent_resistance():
    y = np.linspace(0.0, 1.0, 101)
    weights = interface_control_volume_weights(y, (0.0, 1.0))
    h = y[1] - y[0]
    np.testing.assert_allclose(
        weights, np.r_[h / 2.0, np.full(y.size - 2, h), h / 2.0]
    )
    centers = np.array([-0.5, 0.5, 1.5])
    center_weights = interface_control_volume_weights(centers, (-1.0, 2.0))
    np.testing.assert_allclose(center_weights, 1.0)
    assert equivalent_scalar_resistance(
        centers, np.full(3, 2.0), bounds=(-1.0, 2.0)
    ) == pytest.approx(2.0)


def test_rc_sin_profile_invariants_and_peak():
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



def test_interface_weights_match_solver_dy():
    _, spec, _, spatial, grids, _ = _sample_pair_sin(num_sims=1)
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


def test_forcing_itr_sin_inverse_dispatch_and_determinism():
    adapter = InverseAdapter.from_config(
        {"benchmark": {"name": "forcing_itr_sin"}}
    )
    assert isinstance(adapter, ForcingItrSinAdapter)
    torch.manual_seed(4)
    model = FNO2d(
        modes1=2, modes2=2, width=8, in_channels=5, out_channels=1,
        n_layers=2, cond_static_dim=11, temporal_token_dim=3,
        temporal_hidden=16, forcing_embed_dim=16,
        use_forcing_time_aug=True, s_y_channel=3,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spatial = torch.zeros(3, 12, 12, 5)
    spatial[..., 3] = 1.0
    cond = torch.zeros(3, 11)
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
        "benchmark": np.str_("forcing_itr_sin"),
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
        str(path), benchmark="forcing_itr_sin",
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
            str(path), benchmark="forcing_itr_sin",
            checkpoint_fingerprint="checkpoint",
            dataset_fingerprint="other",
            observation_fingerprint="sensors",
            noise_std_norm=0.05,
        )


def test_validation_calibration_artifact_construction():
    _, spec, _, spatial, grids, _ = _sample_pair_sin(num_sims=3)
    ds = _dataset(spec, spatial, grids)
    ds._split_ids = {
        "train": np.array([0]),
        "val": np.array([1, 2]),
        "test": np.array([], dtype=int),
    }
    adapter = ForcingItrSinAdapter()
    torch.manual_seed(5)
    model = FNO2d(
        modes1=2, modes2=2, width=8, in_channels=5, out_channels=1,
        n_layers=2, cond_static_dim=11, temporal_token_dim=3,
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
    assert np.asarray(calibration["theta_true"]).shape == (2, 2)
    assert "S_R" not in calibration
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
