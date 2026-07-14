import copy
import pickle

import numpy as np
import pytest
import torch

from src.operators.losses import full_bc_physics_loss
from src.operators.train_pino import build_forcing_image
from src.physics.boundary_forcing import (
    A_REF_FLUX,
    BoundaryForcingEvaluator,
    SPATIAL_BUILDERS,
    build_interface_forcing,
    default_ramp_seconds,
    integrate_temporal_ramped_signed,
    ramped_temporal,
    reconstruct_qL,
)
from src.physics.fv_residual import FullBCData, build_cn_geom_per_interface


T_FINAL = 0.2
DT = 0.005


def _sin_params(amplitude=200.0):
    return {
        "A": amplitude, "f": 5.0, "t_on": 0.0, "t_off": T_FINAL,
        "phase": 0.0, "tukey_alpha": 0.5, "rectified": True,
    }


def _record(temporal_family="sin", temporal_params=None):
    return {
        "temporal_family": temporal_family,
        "temporal_params": _sin_params() if temporal_params is None else temporal_params,
        "spatial_family": "uniform",
        "spatial_params": {},
    }


def _build(t_n, t_np1, y_grid=None, ramp=None):
    y_grid = np.linspace(0.0, 1.0, 100) if y_grid is None else y_grid
    ramp = default_ramp_seconds(DT) if ramp is None else ramp
    return build_interface_forcing(
        "sin", _sin_params(), "uniform", {}, y_grid, t_n, t_np1, ramp,
    )


def test_adapter_shapes_dtypes_and_uniform_profile():
    outputs = _build(0.05, 0.055)
    assert len(outputs) == 3
    for value in outputs:
        assert value.shape == (100,)
        assert value.dtype == np.float64
        assert float(value.max() - value.min()) == pytest.approx(0.0, abs=1e-12)


def test_evaluator_copies_parameters_and_grid_matches_points():
    temporal = _sin_params()
    spatial = {"y_c": 0.6, "sigma_y": 0.15}
    forcing = BoundaryForcingEvaluator(
        "sin", temporal, "gaussian", spatial, t_ramp=0.01,
    )
    temporal["A"] = 999.0
    spatial["y_c"] = 0.1
    y = np.linspace(0.0, 1.0, 12)
    t = np.linspace(0.0, T_FINAL, 16)
    grid = forcing.evaluate_grid(y, t)
    yy, tt = np.meshgrid(y, t, indexing="ij")
    points = forcing.evaluate_points(yy, tt)
    np.testing.assert_array_equal(grid, points)
    assert forcing.temporal_params["A"] == 200.0
    assert forcing.spatial_params["y_c"] == 0.6

    restored = pickle.loads(pickle.dumps(forcing))
    np.testing.assert_array_equal(restored.evaluate_grid(y, t), grid)


def test_image_uses_inclusive_axes_and_fixed_normalization():
    y = np.linspace(0.0, 1.0, 96)
    t = np.linspace(0.0, T_FINAL, 256)
    record = _record()
    image = build_forcing_image(
        [record], y, t, A_REF_FLUX, torch.device("cpu"), 0.01,
    )
    forcing = reconstruct_qL(**record, t_ramp=0.01)
    expected = forcing.evaluate_grid(y, t).astype(np.float32) / A_REF_FLUX
    assert image.shape == (1, 1, 96, 256)
    assert image.dtype == torch.float32
    np.testing.assert_array_equal(image[0, 0].numpy(), expected)
    assert t[0] == 0.0 and t[-1] == T_FINAL
    assert y[0] == 0.0 and y[-1] == 1.0


def test_ramp_transition_and_grid_aligned_narrow_pulse_indices():
    params = {"Np": 1, "A_list": [60.0], "t_list": [0.0], "dt_list": [0.2]}
    forcing = reconstruct_qL("pulse_train", params, "uniform", {}, t_ramp=0.01)
    t = np.array([0.0, 0.005, 0.01, 0.015])
    row = forcing.evaluate_grid([0.5], t)[0]
    np.testing.assert_allclose(row, [0.0, 30.0, 60.0, 60.0], rtol=0.0, atol=1e-12)

    narrow = reconstruct_qL(
        "pulse_train",
        {"Np": 1, "A_list": [75.0], "t_list": [0.04], "dt_list": [0.02]},
        "uniform", {}, t_ramp=0.0,
    )
    t_grid = np.linspace(0.0, 0.1, 11)
    values = narrow.evaluate_grid([0.5], t_grid)[0]
    np.testing.assert_array_equal(values[[3, 4, 5, 6]], [0.0, 75.0, 75.0, 0.0])


def test_amplitude_scaling_zero_and_signed_forcing_are_unclipped():
    y = np.linspace(0.0, 1.0, 8)
    t = np.linspace(0.0, 0.1, 16)

    def image(amplitude):
        params = {
            "Np": 1, "A_list": [amplitude], "t_list": [0.0], "dt_list": [0.1],
        }
        return build_forcing_image(
            [_record("pulse_train", params)], y, t, A_REF_FLUX,
            torch.device("cpu"), 0.0,
        )

    base = image(50.0)
    doubled = image(100.0)
    torch.testing.assert_close(doubled, 2.0 * base, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(image(0.0)) == 0
    signed = image(-50.0)
    assert float(signed.min()) < 0.0
    assert float(signed.max()) == 0.0


def test_rendering_is_bitwise_stable_and_rejects_nonpositive_scale():
    record = _record()
    y = np.linspace(0.0, 1.0, 16)
    t = np.linspace(0.0, T_FINAL, 32)
    a = build_forcing_image([record], y, t, 300.0, torch.device("cpu"), 0.01)
    b = build_forcing_image(
        [copy.deepcopy(record)], y.copy(), t.copy(), 300.0,
        torch.device("cpu"), 0.01,
    )
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="a_ref must be > 0"):
        build_forcing_image([record], y, t, 0.0, torch.device("cpu"), 0.01)


def test_flux_adapter_matches_canonical_evaluator_and_previous_formulas():
    ramp = default_ramp_seconds(DT)
    t_n, t_np1 = 0.05, 0.055
    y = np.linspace(0.0, 1.0, 100)
    q_n, q_np1, q_int = _build(t_n, t_np1, y, ramp)
    forcing = reconstruct_qL("sin", _sin_params(), "uniform", {}, ramp)
    np.testing.assert_array_equal(q_n, forcing.evaluate_points(y, t_n))
    np.testing.assert_array_equal(q_np1, forcing.evaluate_points(y, t_np1))
    np.testing.assert_array_equal(q_int, forcing.integral(y, t_n, t_np1))

    old_a = ramped_temporal("sin", _sin_params(), ramp)
    old_s = SPATIAL_BUILDERS["uniform"](y)
    np.testing.assert_allclose(q_n, old_a(t_n) * old_s, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(q_np1, old_a(t_np1) * old_s, rtol=0.0, atol=0.0)
    old_int = integrate_temporal_ramped_signed(
        "sin", _sin_params(), t_n, t_np1, ramp,
    ) * old_s
    np.testing.assert_allclose(q_int, old_int, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("t_n,t_np1", [(0.0, DT), (0.05, 0.055), (0.1, 0.11)])
def test_qL_int_is_exact_signed_integral(t_n, t_np1):
    ramp = default_ramp_seconds(DT)
    _, _, q_int = _build(t_n, t_np1, ramp=ramp)
    a_fn = ramped_temporal("sin", _sin_params(), ramp)
    times = np.linspace(t_n, t_np1, 20001)
    reference = np.trapezoid([a_fn(float(value)) for value in times], times)
    assert float(q_int[0]) == pytest.approx(float(reference), rel=1e-4, abs=1e-8)


def test_complete_fv_loss_is_invariant_to_forcing_reconstruction_refactor():
    x = np.linspace(0.0, 1.0, 20)
    y = np.linspace(0.0, 1.0, 20)
    t_n, t_np1, ramp = 0.05, 0.055, 0.01
    params = _sin_params()
    q_new = build_interface_forcing(
        "sin", params, "uniform", {}, y, t_n, t_np1, ramp,
    )
    a_fn = ramped_temporal("sin", params, ramp)
    s_y = SPATIAL_BUILDERS["uniform"](y)
    q_old = (
        a_fn(t_n) * s_y,
        a_fn(t_np1) * s_y,
        integrate_temporal_ramped_signed("sin", params, t_n, t_np1, ramp) * s_y,
    )
    generator = torch.Generator().manual_seed(7)
    T_n = torch.randn(2, 20, 20, generator=generator)
    T_np1 = torch.randn(2, 20, 20, generator=generator)
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0,
        np.array([(x[8] + x[9]) / 2.0, (x[10] + x[11]) / 2.0]),
        np.array([0.2, 0.7]),
        DT, sigma_global=3.0, dtype=torch.float32,
    )

    def losses(q_values):
        bc = FullBCData(
            T_right_tilde=torch.tensor(0.0),
            qL_n=torch.tensor(np.stack([q_values[0], q_values[0]])).float(),
            qL_np1=torch.tensor(np.stack([q_values[1], q_values[1]])).float(),
            qL_int=torch.tensor(np.stack([q_values[2], q_values[2]])).float(),
        )
        return full_bc_physics_loss(
            T_n, T_np1, geom, bc, per_sample=True, interface_band=True,
        )

    before, after = losses(q_old), losses(q_new)
    keys = (
        "phys_interior_mse", "phys_interface_band_mse",
        "phys_left_neumann_mse", "phys_topbot_adiabatic_mse",
        "phys_right_dirichlet_mse", "physics_loss_weighted",
        "physics_loss_allcell_mean",
    )
    for key in keys:
        torch.testing.assert_close(before[key], after[key], rtol=0.0, atol=0.0)
