import numpy as np
import pytest
import torch

from scripts.diagnose_fv_one_step_cvit import (
    build_interval_forcing_images,
    conservative_energy_storage_projection,
    interface_flux_head_physics_terms,
    interface_flux_agreement_metrics,
    left_energy_interface_flux_closure,
    loss_gradient_geometry,
    one_step_interface_trace_terms,
    sample_stratified_forcing_probes,
    state_conditioned_spatial,
    two_sided_energy_interface_flux_closure,
)
from src.physics.boundary_forcing import build_interface_forcing
from src.physics.fv_residual import build_cn_geom_per_interface


def _params():
    return {
        "temporal_family": "pulse_train",
        "temporal_params": {
            "Np": 1,
            "A_list": [250.0],
            "t_list": [0.04],
            "dt_list": [0.08],
        },
        "spatial_family": "uniform",
        "spatial_params": {},
    }


def test_interval_image_is_exact_average_flux():
    params = _params()
    y = np.linspace(0.0, 1.0, 12)
    times = np.array([0.0, 0.005, 0.01, 0.015])
    images = build_interval_forcing_images(
        params,
        y,
        times,
        t_ramp=0.01,
        q_ref=300.0,
        image_time_points=8,
    )
    assert images.shape == (3, 1, 12, 8)
    for n in range(3):
        _, _, q_integral = build_interface_forcing(
            params["temporal_family"],
            params["temporal_params"],
            params["spatial_family"],
            params["spatial_params"],
            y,
            times[n],
            times[n + 1],
            0.01,
        )
        expected = q_integral / ((times[n + 1] - times[n]) * 300.0)
        np.testing.assert_allclose(images[n, 0, :, 0], expected)
        np.testing.assert_allclose(
            images[n, 0], np.broadcast_to(expected[:, None], (len(y), 8))
        )


def test_stratified_probe_sampling_is_deterministic_complete_and_disjoint():
    kwargs = {
        "per_combination": 1,
        "dt": 0.005,
        "t_final": 0.3,
        "y_bounds": (0.0, 1.0),
    }
    train = sample_stratified_forcing_probes(1729, prefix="train", **kwargs)
    repeat = sample_stratified_forcing_probes(1729, prefix="train", **kwargs)
    holdout = sample_stratified_forcing_probes(2718, prefix="holdout", **kwargs)

    assert train == repeat
    assert len(train) == len(holdout) == 16
    assert set(train).isdisjoint(holdout)
    assert {
        (params["temporal_family"], params["spatial_family"])
        for params in train.values()
    } == {
        (temporal, spatial)
        for temporal in ("sin", "exp", "pulse_train", "exp_train")
        for spatial in ("uniform", "patch", "gaussian", "triangle")
    }
    assert all(0.05 <= params["R_c"] <= 1.0 for params in train.values())
    assert any(
        train_case["temporal_params"] != holdout_case["temporal_params"]
        or train_case["spatial_params"] != holdout_case["spatial_params"]
        or train_case["R_c"] != holdout_case["R_c"]
        for train_case, holdout_case in zip(train.values(), holdout.values())
    )


def test_state_is_live_first_spatial_channel():
    state = torch.randn(2, 10, 10)
    material = torch.randn(1, 2, 10, 10)
    spatial = state_conditioned_spatial(state, material.expand(2, -1, -1, -1))
    assert spatial.shape == (2, 3, 10, 10)
    assert torch.equal(spatial[:, 0], state)
    assert torch.equal(spatial[:, 1:], material.expand(2, -1, -1, -1))


def test_one_step_trace_terms_are_zero_for_piecewise_linear_contact_field():
    x = np.arange(8, dtype=np.float64) + 0.5
    y = np.arange(8, dtype=np.float64) + 0.5
    interface = 4.0
    resistance = 0.2
    heat_flux = 1.7
    right_trace = 301.0
    left_trace = right_trace + resistance * heat_flux
    physical = np.where(
        x[:, None] < interface,
        left_trace + heat_flux * (interface - x[:, None]) / 2.0,
        right_trace + heat_flux * (interface - x[:, None]),
    )
    normalized = torch.from_numpy(
        (np.broadcast_to(physical, (8, 8)).copy() - 300.0) / 5.0
    ).to(torch.float64)[None]
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0, [interface], [resistance], 0.01,
        sigma_global=5.0, device=torch.device("cpu"), dtype=torch.float64,
    )
    terms = one_step_interface_trace_terms(
        normalized, geom, x, resistance,
        k_left=2.0, k_right=1.0, sigma=5.0, q_ref=3.0,
    )
    assert float(terms["flux_rms"]) < 1e-11
    assert float(terms["contact_rms"]) < 1e-11
    assert float(terms["q_series_rms"]) == pytest.approx(heat_flux)
    assert float(terms["jump_rms_K"]) == pytest.approx(resistance * heat_flux)


def test_gradient_geometry_reports_norms_cosines_without_populating_grad():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    variational = parameter[0] + parameter[1]
    flux = parameter[0] - parameter[1]
    contact = -variational
    diagnostics = loss_gradient_geometry(
        {
            "variational": variational,
            "interface_flux": flux,
            "interface_contact": contact,
        },
        [parameter],
    )
    assert parameter.grad is None
    assert diagnostics["grad_norm_variational"] == pytest.approx(np.sqrt(2.0))
    assert diagnostics["grad_norm_interface_flux"] == pytest.approx(np.sqrt(2.0))
    assert diagnostics["grad_cos_variational_interface_flux"] == pytest.approx(0.0)
    assert diagnostics["grad_cos_variational_interface_contact"] == pytest.approx(-1.0)


def test_flux_head_left_balance_uses_exact_prescribed_flux_integral():
    x = np.arange(8, dtype=np.float64) + 0.5
    y = np.arange(8, dtype=np.float64) + 0.5
    dt = 0.01
    q_left = 2.5
    q_ref = 10.0
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0, [4.0], [0.2], dt,
        sigma_global=5.0, device=torch.device("cpu"), dtype=torch.float64,
    )
    state = torch.zeros(1, 8, 8, dtype=torch.float64)
    q_head = torch.full((1, 8), 2.0 * q_left / q_ref, dtype=torch.float64)
    terms = interface_flux_head_physics_terms(
        state,
        state,
        q_head,
        geom,
        torch.full((1, 8), q_left * dt, dtype=torch.float64),
        right_value=0.0,
        q_ref=q_ref,
        sigma=5.0,
    )
    assert float(terms["left_energy_rms"]) < 1e-12
    assert float(terms["q_head_rms"]) == pytest.approx(2.0 * q_left)
    assert float(terms["series_rms"]) == pytest.approx(2.0 * q_left / q_ref)


def test_time_resolved_flux_agreement_does_not_confuse_amplitude_with_phase():
    truth = np.array([0.0, 1.0, 2.0, 1.0])
    aligned = interface_flux_agreement_metrics(truth, truth, prefix="q")
    shifted = interface_flux_agreement_metrics(
        np.roll(truth, 1), truth, prefix="q"
    )
    assert aligned["q_rmse"] == 0.0
    assert aligned["q_correlation"] == pytest.approx(1.0)
    assert shifted["q_rmse"] > 0.0
    assert shifted["q_correlation"] < 1.0


def test_left_energy_flux_closure_makes_enriched_balance_exact_and_is_live():
    x = np.arange(8, dtype=np.float64) + 0.5
    y = np.arange(8, dtype=np.float64) + 0.5
    dt = 0.01
    resistance = 0.2
    sigma = 5.0
    q_ref = 10.0
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0, [4.0], [resistance], dt,
        sigma_global=sigma, device=torch.device("cpu"), dtype=torch.float64,
    )
    state = torch.zeros(1, 8, 8, dtype=torch.float64)
    smooth = torch.zeros_like(state, requires_grad=True)
    q_left_integral = torch.full((1, 8), 2.5 * dt, dtype=torch.float64)
    flux = left_energy_interface_flux_closure(
        state, smooth, geom, q_left_integral,
        resistance=resistance, sigma=sigma,
    )
    prediction = smooth.clone()
    prediction[:, :4] = prediction[:, :4] + resistance * flux[:, None, None] / sigma
    terms = interface_flux_head_physics_terms(
        state,
        prediction,
        (flux / q_ref)[:, None].expand(-1, 8),
        geom,
        q_left_integral,
        right_value=0.0,
        q_ref=q_ref,
        sigma=sigma,
    )
    assert float(terms["left_energy_rms"]) < 1e-12
    gradient = torch.autograd.grad(flux.sum(), smooth)[0]
    assert float(gradient[:, :4].abs().sum()) > 0.0


def test_two_sided_flux_closure_recovers_flux_when_both_balances_agree():
    x = np.arange(8, dtype=np.float64) + 0.5
    y = np.arange(8, dtype=np.float64) + 0.5
    dt = 0.01
    resistance = 0.2
    sigma = 5.0
    q_ref = 10.0
    target_flux = 2.5
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0, [4.0], [resistance], dt,
        sigma_global=sigma, device=torch.device("cpu"), dtype=torch.float64,
    )
    state = torch.zeros(1, 8, 8, dtype=torch.float64)
    smooth = torch.zeros_like(state, requires_grad=True)
    dy = geom.dy
    height = dy.sum()
    capacity = geom.rho_cp[None] * geom.dx[:, :, None] * dy[None, None]
    right_capacity = capacity[0, 4:7].sum()
    right_conductance = geom.G_x[0, 6, 0]
    right_value = (
        0.5 * dt * height * target_flux
        / (sigma * right_capacity + 0.5 * dt * height * right_conductance * sigma)
    )
    smooth = smooth.clone()
    smooth[:, 4:7] = right_value
    left_capacity = capacity[0, :4].sum()
    left_coefficient = resistance * left_capacity + 0.5 * dt * height
    injected_total = left_coefficient * target_flux
    q_left_integral = torch.full(
        (1, 8), injected_total / height, dtype=torch.float64
    )
    flux = two_sided_energy_interface_flux_closure(
        state,
        smooth,
        geom,
        q_left_integral,
        resistance=resistance,
        right_value=0.0,
        q_ref=q_ref,
        sigma=sigma,
    )
    assert float(flux) == pytest.approx(target_flux, rel=1e-12, abs=1e-12)
    prediction = smooth.clone()
    prediction[:, :4] += resistance * flux[:, None, None] / sigma
    terms = interface_flux_head_physics_terms(
        state,
        prediction,
        (flux / q_ref)[:, None].expand(-1, 8),
        geom,
        q_left_integral,
        right_value=0.0,
        q_ref=q_ref,
        sigma=sigma,
    )
    assert float(terms["left_energy_rms"]) < 1e-12
    assert float(terms["right_energy_rms"]) < 1e-12


def test_conservative_storage_projection_satisfies_both_layer_balances():
    x = np.arange(8, dtype=np.float64) + 0.5
    y = np.arange(8, dtype=np.float64) + 0.5
    dt = 0.01
    resistance = 0.2
    sigma = 5.0
    q_ref = 10.0
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0, [4.0], [resistance], dt,
        sigma_global=sigma, device=torch.device("cpu"), dtype=torch.float64,
    )
    state = torch.zeros(1, 8, 8, dtype=torch.float64)
    smooth = torch.zeros_like(state, requires_grad=True)
    q_left_integral = torch.full((1, 8), 2.5 * dt, dtype=torch.float64)
    flux, projected, correction, implied_flux_gap = (
        conservative_energy_storage_projection(
        state,
        smooth,
        geom,
        q_left_integral,
        resistance=resistance,
        right_value=0.0,
        sigma=sigma,
        )
    )
    assert torch.equal(correction[:, :5], torch.zeros_like(correction[:, :5]))
    assert torch.equal(correction[:, 6:], torch.zeros_like(correction[:, 6:]))
    prediction = projected.clone()
    prediction[:, :4] += resistance * flux[:, None, None] / sigma
    terms = interface_flux_head_physics_terms(
        state,
        prediction,
        (flux / q_ref)[:, None].expand(-1, 8),
        geom,
        q_left_integral,
        right_value=0.0,
        q_ref=q_ref,
        sigma=sigma,
    )
    assert float(terms["left_energy_rms"]) < 1e-12
    assert float(terms["right_energy_rms"]) < 1e-12
    assert torch.allclose(implied_flux_gap, torch.zeros_like(implied_flux_gap), atol=1e-5)
    gradient = torch.autograd.grad(projected.sum(), smooth, retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    penalty_gradient = torch.autograd.grad(correction.square().mean(), smooth)[0]
    assert float(penalty_gradient.abs().sum()) > 0.0
