import csv
import math

import numpy as np
import pytest
import torch

from data.dataset import split_sim_ids
from src.operators.losses import region_balanced_fv_rate_loss
import src.operators.train_pino as train_pino_module
from scripts.diagnose_supervised_overfit import _forcing_supervised_metrics
from src.operators.train_pino import (
    ForcingConstraintController,
    load_interface_cvit_checkpoint,
    run_config_seeds_pino,
    run_one_seed_forcing_interface_pino,
)
from src.operators.cvit import InterfaceCViT
from src.operators.train_pino import (
    HybridForcingRNGs,
    _decode_derivatives,
    _forcing_diagnostic_gradient_metrics,
    _forcing_layered_batch_losses,
    _forcing_probe_metrics,
    _sample_online_layered_forcing_params,
    _sample_hybrid_bulk,
    _sample_lattice_intervals,
)
from src.physics.fv_residual import (
    FullBCData,
    block_energy_residual,
    build_cn_geom_per_interface,
    full_bc_cn_residual,
    full_bc_cn_residual_rate,
    interface_residual_rate,
    interface_trace_constraint_residuals,
    region_balanced_fv_rate_residual,
)


def _geometry(batch=2, nx=8, ny=8, dt=0.05, sigma=2.5):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    interface = 0.5 * (x[nx // 2 - 1] + x[nx // 2])
    geom = build_cn_geom_per_interface(
        x, y, 2.0, 1.0, np.full(batch, interface),
        np.linspace(0.2, 0.8, batch), dt,
        sigma_global=sigma, device="cpu", dtype=torch.float64,
    )
    return x, y, interface, geom


def _fields_and_bc(batch=2, nx=8, ny=8):
    generator = torch.Generator().manual_seed(3)
    T_n = torch.randn(batch, nx, ny, generator=generator, dtype=torch.float64)
    T_np1 = torch.randn(batch, nx, ny, generator=generator, dtype=torch.float64)
    qn = torch.randn(batch, ny, generator=generator, dtype=torch.float64)
    qnp1 = torch.randn(batch, ny, generator=generator, dtype=torch.float64)
    qint = 0.05 * 0.5 * (qn + qnp1)
    bc = FullBCData(
        T_right_tilde=torch.tensor(-0.2, dtype=torch.float64),
        qL_n=qn, qL_np1=qnp1, qL_int=qint,
    )
    return T_n, T_np1, bc


def test_rate_residual_is_exact_increment_residual_divided_by_dt():
    _, _, _, geom = _geometry()
    T_n, T_np1, bc = _fields_and_bc()
    increment = full_bc_cn_residual(
        T_n, T_np1, geom, bc, dirichlet_both_ends=True, keep_batch=True
    )
    rate = full_bc_cn_residual_rate(
        T_n, T_np1, geom, bc, dirichlet_both_ends=True, keep_batch=True
    )
    assert torch.allclose(geom.dt * rate["interior"], increment["interior"])
    assert torch.allclose(
        geom.dt * rate["topbot_adiabatic"], increment["topbot_adiabatic"]
    )
    assert torch.allclose(
        geom.dt * rate["left_neumann"], increment["left_neumann"]
    )
    assert torch.equal(rate["right_dirichlet"], increment["right_dirichlet"])
    assert torch.equal(rate["right_dirichlet_n"], increment["right_dirichlet_n"])


def test_interface_local_and_full_grid_values_and_gradients_match():
    _, _, _, geom = _geometry()
    face = int(geom.face_idx[0])
    columns = torch.tensor([face - 1, face, face + 1, face + 2])
    generator = torch.Generator().manual_seed(7)
    T_n_full = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    T_np1_full = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    full = interface_residual_rate(T_n_full, T_np1_full, geom)
    full_loss = full.square().mean()
    full_grads = torch.autograd.grad(full_loss, (T_n_full, T_np1_full))

    T_n_local_source = T_n_full.detach().clone().requires_grad_(True)
    T_np1_local_source = T_np1_full.detach().clone().requires_grad_(True)
    local = interface_residual_rate(
        T_n_local_source[:, columns], T_np1_local_source[:, columns], geom
    )
    local_loss = local.square().mean()
    local_grads = torch.autograd.grad(
        local_loss, (T_n_local_source, T_np1_local_source)
    )
    assert full.shape == (2, 2, 8)
    assert torch.allclose(local, full, atol=1e-12, rtol=1e-12)
    assert torch.allclose(local_loss, full_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(local_grads[0], full_grads[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(local_grads[1], full_grads[1], atol=1e-12, rtol=1e-12)

    basis_n = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64)
    basis_np1 = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64)
    theta_full = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    theta_local = theta_full.detach().clone().requires_grad_(True)
    parameter_loss_full = interface_residual_rate(
        T_n_full.detach() + theta_full * basis_n,
        T_np1_full.detach() + theta_full * basis_np1,
        geom,
    ).square().mean()
    parameter_loss_local = interface_residual_rate(
        (T_n_full.detach() + theta_local * basis_n)[:, columns],
        (T_np1_full.detach() + theta_local * basis_np1)[:, columns],
        geom,
    ).square().mean()
    parameter_grad_full = torch.autograd.grad(parameter_loss_full, theta_full)[0]
    parameter_grad_local = torch.autograd.grad(parameter_loss_local, theta_local)[0]
    assert torch.allclose(
        parameter_grad_local, parameter_grad_full, atol=1e-12, rtol=1e-12
    )


def test_region_partition_includes_wall_interface_cells_and_recombines():
    _, _, _, geom = _geometry()
    T_n, T_np1, bc = _fields_and_bc()
    legacy = full_bc_cn_residual_rate(T_n, T_np1, geom, bc, keep_batch=True)
    regions = region_balanced_fv_rate_residual(
        T_n, T_np1, geom, bc, keep_batch=True
    )
    assert regions["interface"].shape == (2, 2, 8)
    assert regions["interior"].shape == (2, 4, 6)
    assert regions["left_neumann"].shape == (2, 8)
    assert regions["top_adiabatic"].shape == (2, 4)
    assert regions["bottom_adiabatic"].shape == (2, 4)
    assert sum(
        value[0].numel()
        for key, value in regions.items()
        if key in {
            "interior", "interface", "left_neumann",
            "top_adiabatic", "bottom_adiabatic",
        }
    ) == 7 * 8
    old_energy = torch.cat([
        legacy["interior"].flatten(1),
        legacy["topbot_adiabatic"].flatten(1),
        legacy["left_neumann"].flatten(1),
    ], dim=1)
    new_energy = torch.cat([
        regions["interior"].flatten(1),
        regions["interface"].flatten(1),
        regions["top_adiabatic"].flatten(1),
        regions["bottom_adiabatic"].flatten(1),
        regions["left_neumann"].flatten(1),
    ], dim=1)
    assert new_energy.shape == old_energy.shape
    assert torch.allclose(
        new_energy.square().sum(dim=1), old_energy.square().sum(dim=1),
        atol=1e-10, rtol=1e-12,
    )

    loss = region_balanced_fv_rate_loss(T_n, T_np1, geom, bc, t_ref=0.3)
    assert torch.allclose(
        loss["topbot_adiabatic_per_sample"],
        0.5 * (
            loss["top_adiabatic_per_sample"]
            + loss["bottom_adiabatic_per_sample"]
        ),
    )


def test_rate_boundary_uses_physical_flux_and_normalized_dirichlet():
    _, _, _, geom = _geometry(batch=1, sigma=4.0)
    zeros = torch.zeros(1, 8, 8, dtype=torch.float64)
    qint = torch.full((1, 8), 0.1, dtype=torch.float64)
    target_kelvin = 310.0
    mu, sigma = 300.0, 4.0
    target_tilde = (target_kelvin - mu) / sigma
    bc = FullBCData(
        T_right_tilde=torch.tensor(target_tilde, dtype=torch.float64),
        qL_n=torch.zeros(1, 8, dtype=torch.float64),
        qL_np1=torch.zeros(1, 8, dtype=torch.float64),
        qL_int=qint,
    )
    rate = full_bc_cn_residual_rate(zeros, zeros, geom, bc, keep_batch=True)
    expected_left = -2.0 * qint / (
        geom.dt * geom.rho_cp[0][None, :] * geom.hx * sigma
    )
    assert torch.allclose(rate["left_neumann"], expected_left)
    assert torch.allclose(
        rate["right_dirichlet"], torch.full((1, 8), -target_tilde, dtype=torch.float64)
    )


class _CoordinateDecoder(torch.nn.Module):
    def forward(self, latent, coords, normalized_time):
        return (
            coords[..., 0:1].square()
            + 3.0 * coords[..., 1:2].square()
            + 2.0 * normalized_time
        )


def test_physical_coordinate_and_time_chain_rules_are_retained():
    model = InterfaceCViT(
        spatial_in_ch=3, forcing_in_ch=1, emb_dim=8, patch_size=4,
        grid_size=(8, 8), forcing_patch_size=4, forcing_grid_size=(8, 8),
        depth_enc=1, depth_dec=1, num_heads=2, param_hidden=8,
        hard_right_dirichlet=False, t_final=0.25,
    )
    model.decoder = _CoordinateDecoder()
    latent = torch.zeros(1, 1, 8)
    coords = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.6]]], requires_grad=True
    )
    time_query = torch.tensor([[[0.05], [0.15]]], requires_grad=True)
    _, grad_xy, grad_t, grad_xx, grad_yy = _decode_derivatives(
        model, latent, coords, time_query, second_order=True
    )
    assert torch.allclose(grad_xy[..., 0:1], 2.0 * coords[..., 0:1])
    assert torch.allclose(grad_xy[..., 1:2], 6.0 * coords[..., 1:2])
    assert torch.allclose(grad_t, torch.full_like(grad_t, 8.0))
    assert torch.allclose(grad_xx, torch.full_like(grad_xx, 2.0))
    assert torch.allclose(grad_yy, torch.full_like(grad_yy, 6.0))


def test_piecewise_linear_contact_solution_zeroes_all_interface_rows():
    x, _, interface, geom = _geometry(batch=1, sigma=5.0)
    heat_flux = 1.7
    resistance = 0.2
    T_right = 300.0
    right_trace = T_right + heat_flux * (1.0 - interface)
    left_trace = right_trace + resistance * heat_flux
    physical = np.where(
        x[:, None] < interface,
        left_trace + heat_flux * (interface - x[:, None]) / 2.0,
        right_trace + heat_flux * (interface - x[:, None]),
    )
    physical = np.broadcast_to(physical, (8, 8)).copy()
    normalized = torch.from_numpy((physical - 300.0) / 5.0).to(torch.float64)[None]
    rate = interface_residual_rate(normalized, normalized, geom)
    assert rate.shape == (1, 2, 8)
    assert float(rate.abs().max()) < 1e-12


def test_block_energy_constant_field_exposes_exact_prescribed_injection():
    _, _, _, geom = _geometry(batch=1, sigma=5.0)
    zeros = torch.zeros(1, 8, 8, dtype=torch.float64)
    qint = torch.full((1, 8), 0.05 * 2.0, dtype=torch.float64)
    blocks = block_energy_residual(
        zeros, zeros, geom, qint,
        x_blocks_per_layer=2, y_blocks=2, q_ref=3.0,
    )
    assert blocks["all"].shape == (1, 8)
    assert blocks["interface"].shape == (1, 4)
    assert blocks["far"].shape == (1, 4)
    assert float(blocks["all"].abs().max()) > 0.0
    assert torch.count_nonzero(blocks["numerator"]) == 2
    expected_injection = -(qint * geom.dy[None]).sum()
    assert blocks["numerator"].sum() == pytest.approx(float(expected_injection))


def test_block_energy_zero_flux_constant_field_is_exact_and_partitions_blocks():
    _, _, _, geom = _geometry(batch=1, sigma=5.0)
    zeros = torch.zeros(1, 8, 8, dtype=torch.float64)
    blocks = block_energy_residual(
        zeros, zeros, geom, torch.zeros(1, 8, dtype=torch.float64),
        x_blocks_per_layer=2, y_blocks=2, q_ref=3.0,
    )
    assert torch.equal(blocks["all"], torch.zeros_like(blocks["all"]))
    assert int(blocks["interface_mask"].sum()) == 4
    assert sorted(set(blocks["block_x"].tolist())) == [0, 1, 2, 3]
    assert sorted(set(blocks["block_y"].tolist())) == [0, 1]


def test_quadratic_interface_constraints_accept_piecewise_linear_contact_field():
    x, _, interface, geom = _geometry(batch=1, sigma=5.0)
    heat_flux = 1.7
    resistance = 0.2
    right_trace = 301.0
    left_trace = right_trace + resistance * heat_flux
    physical = np.where(
        x[:, None] < interface,
        left_trace + heat_flux * (interface - x[:, None]) / 2.0,
        right_trace + heat_flux * (interface - x[:, None]),
    )
    physical = np.broadcast_to(physical, (8, 8)).copy()
    normalized = torch.from_numpy((physical - 300.0) / 5.0).to(torch.float64)[None]
    residuals = interface_trace_constraint_residuals(
        normalized, normalized, geom, x, [resistance],
        k_left=2.0, k_right=1.0, sigma_global=5.0, q_ref=3.0,
    )
    assert float(residuals["flux"].abs().max()) < 1e-11
    assert float(residuals["contact"].abs().max()) < 1e-11
    assert torch.allclose(
        residuals["q_series"], torch.full_like(residuals["q_series"], heat_flux),
        atol=1e-11, rtol=1e-11,
    )


def test_interface_constraints_detect_wrong_jump_and_reach_both_sides():
    x, _, _, geom = _geometry(batch=1, sigma=5.0)
    field = torch.zeros(1, 8, 8, dtype=torch.float64, requires_grad=True)
    perturbed = field.clone()
    perturbed[:, 3] = 0.4
    residuals = interface_trace_constraint_residuals(
        perturbed, perturbed, geom, x, [0.2],
        k_left=2.0, k_right=1.0, sigma_global=5.0, q_ref=3.0,
    )
    loss = residuals["flux"].square().mean() + residuals["contact"].square().mean()
    gradient = torch.autograd.grad(loss, field)[0]
    assert float(loss) > 0.0
    assert float(gradient[:, :4].abs().sum()) > 0.0
    assert float(gradient[:, 4:].abs().sum()) > 0.0


def test_backend_specific_draws_do_not_perturb_common_rng_streams():
    first = HybridForcingRNGs.create(19, torch.device("cpu"))
    second = HybridForcingRNGs.create(19, torch.device("cpu"))
    _ = torch.rand(200, generator=first.bulk_collocation)
    _ = torch.rand(200, generator=first.boundary_collocation)
    assert np.array_equal(
        first.forcing_selection.integers(0, 100, size=20),
        second.forcing_selection.integers(0, 100, size=20),
    )
    assert np.array_equal(
        first.contact_resistance.integers(0, 100, size=20),
        second.contact_resistance.integers(0, 100, size=20),
    )
    assert np.array_equal(
        first.interval_selection.integers(0, 100, size=20),
        second.interval_selection.integers(0, 100, size=20),
    )


def test_layered_training_problems_are_sampled_online_without_saved_ic():
    rngs = HybridForcingRNGs.create(23, torch.device("cpu"))
    params = _sample_online_layered_forcing_params(
        rngs,
        512,
        dt=0.005,
        t_final=0.3,
        y_bounds=(0.0, 1.0),
        temporal_window={
            "t_on": 0.0, "t_off": 0.2, "phase": 0.0, "tukey_alpha": 0.5,
        },
    )
    assert {record["temporal_family"] for record in params} == {
        "sin", "exp", "pulse_train", "exp_train",
    }
    assert {record["spatial_family"] for record in params} == {
        "uniform", "patch", "gaussian", "triangle",
    }
    assert all("T0" not in record for record in params)
    assert all(record["interface_x"] == pytest.approx(0.5) for record in params)
    assert all(0.05 <= record["R_c"] <= 1.0 for record in params)


def test_second_derivative_loss_reaches_all_interface_cvit_paths():
    model = InterfaceCViT(
        spatial_in_ch=3, forcing_in_ch=1, emb_dim=8, patch_size=4,
        grid_size=(8, 8), forcing_patch_size=4, forcing_grid_size=(8, 8),
        depth_enc=1, depth_dec=1, num_heads=2, param_hidden=8,
        hard_right_dirichlet=True, t_final=0.3,
    )
    generator = torch.Generator().manual_seed(5)
    spatial = torch.randn(2, 3, 8, 8, generator=generator)
    forcing = torch.randn(2, 1, 8, 8, generator=generator)
    scalars = torch.rand(2, 2, generator=generator)
    latent = model.encode(spatial, forcing, scalars)
    coords, time_query = _sample_hybrid_bulk(
        2, 8, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0),
        exclusion=(3.0 / 7.0, 4.0 / 7.0), t_final=0.3,
        device=torch.device("cpu"), generator=generator,
    )
    _, _, grad_t, grad_xx, grad_yy = _decode_derivatives(
        model, latent, coords, time_query, second_order=True
    )
    loss = (0.3 * (grad_t - grad_xx - grad_yy)).square().mean()
    named = [(name, parameter) for name, parameter in model.named_parameters()]
    gradients = torch.autograd.grad(
        loss, [parameter for _, parameter in named], allow_unused=True
    )
    groups = {
        "spatial_encoder": 0.0,
        "forcing_encoder": 0.0,
        "param_encoder": 0.0,
        "decoder": 0.0,
    }
    for (name, _), gradient in zip(named, gradients):
        if gradient is None:
            continue
        assert torch.isfinite(gradient).all(), name
        for prefix in groups:
            if name.startswith(prefix):
                groups[prefix] += float(gradient.square().sum())
    assert all(value > 0.0 for value in groups.values()), groups


@pytest.mark.slow
def test_interface_cvit_supervised_piecewise_jump_representability():
    torch.manual_seed(0)
    model = InterfaceCViT(
        spatial_in_ch=3, forcing_in_ch=1, emb_dim=8, patch_size=4,
        grid_size=(8, 8), forcing_patch_size=4, forcing_grid_size=(8, 8),
        depth_enc=1, depth_dec=1, num_heads=2, param_hidden=8,
        hard_right_dirichlet=True, t_final=0.3,
    )
    spatial = torch.zeros(1, 3, 8, 8)
    forcing = torch.zeros(1, 1, 8, 8)
    scalars = torch.tensor([[0.5, 0.3]])
    x = torch.linspace(0.0, 1.0, 8)
    gx, gy = torch.meshgrid(x, x, indexing="ij")
    coords = torch.stack((gx.flatten(), gy.flatten()), dim=-1).unsqueeze(0)
    time_query = torch.full((1, 64, 1), 0.1)
    right_trace = 0.1
    left_trace = 0.18
    target = torch.where(
        coords[..., 0:1] < 0.5,
        left_trace + (0.5 - coords[..., 0:1]) / 10.0,
        right_trace + (0.5 - coords[..., 0:1]) / 5.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(201):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(spatial, coords, time_query, forcing, scalars)
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()
    prediction = model(
        spatial, coords, time_query, forcing, scalars
    ).detach().view(8, 8)
    target_grid = target.view(8, 8)
    assert float((prediction - target_grid).square().mean().sqrt()) < 0.01
    predicted_jump = float((prediction[3] - prediction[4]).mean())
    target_jump = float((target_grid[3] - target_grid[4]).mean())
    assert abs(predicted_jump - target_jump) < 0.02


class _PiecewiseSteadyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = torch.nn.Parameter(
            torch.tensor([0.1, -0.1, 0.05, 0.1, -0.1, 0.05, 0.1])
        )

    def encode(self, spatial, forcing, scalars):
        return torch.zeros(spatial.shape[0], 1, 1, device=spatial.device)

    def decode(self, latent, coords, time_query):
        x = coords[..., 0:1]
        left_distance = 0.5 - x
        right_distance = 1.0 - x
        c = self.coefficients
        left = (
            c[0] + c[1] * left_distance + c[2] * left_distance.square()
            + c[6] * time_query
        )
        right = (
            c[3] * right_distance + c[4] * right_distance.square()
            + c[5] * right_distance.pow(3) + c[6] * time_query * right_distance
        )
        return torch.where(x < 0.5, left, right)


def _assert_rng_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_rng_state_equal(left[key], right[key])
    else:
        assert left == right


def _toy_probe_outputs(forcing_sensitive: bool):
    pred = np.full((4, 2, 1, 1), 300.0, dtype=np.float64)
    if forcing_sensitive:
        pred[0, :, 0, 0] = [301.0, 302.0]
    pred[1, :, 0, 0] = [300.5, 301.0]
    pred[3, :, 0, 0] = [300.2, 300.4]
    jumps = np.zeros((4, 2, 2, 1), dtype=np.float64)
    jumps[:, :, 0, 0] = pred[:, :, 0, 0] + 0.1
    jumps[:, :, 1, 0] = pred[:, :, 0, 0] - 0.1
    truth0 = np.asarray([[[320.0]], [[340.0]]], dtype=np.float64)
    truth1 = np.asarray([[[315.0]], [[330.0]]], dtype=np.float64)
    truth3 = np.asarray([[[305.0]], [[310.0]]], dtype=np.float64)
    truth_jump0 = np.asarray([[[321.0], [319.0]], [[342.0], [338.0]]])
    truth_jump1 = np.asarray([[[316.0], [314.0]], [[332.0], [328.0]]])
    truth_jump3 = np.asarray([[[306.0], [304.0]], [[312.0], [308.0]]])
    return {
        "sparse_prediction": pred,
        "jump_prediction": jumps,
        "truth_sparse": [truth0, truth1, None, truth3],
        "truth_jump": [truth_jump0, truth_jump1, None, truth_jump3],
        "zero_flags": np.asarray([False, False, True, False]),
    }


def test_fixed_probe_response_ratio_detects_forcing_sensitive_toy():
    outputs = _toy_probe_outputs(forcing_sensitive=True)
    metrics = _forcing_probe_metrics(
        outputs, {"sparse_prediction": outputs["sparse_prediction"].copy()}
    )
    assert metrics["drift_from_init_K"] == pytest.approx(0.0)
    assert metrics["forcing_sensitivity_K"] > 0.0
    assert metrics["forcing_response_ratio"] > 0.0


def test_fixed_probe_response_ratio_is_zero_for_forcing_independent_toy():
    outputs = _toy_probe_outputs(forcing_sensitive=False)
    metrics = _forcing_probe_metrics(outputs, None)
    assert metrics["forcing_sensitivity_K"] == pytest.approx(0.0)
    assert metrics["forcing_response_ratio"] == pytest.approx(0.0)


def test_fixed_gradient_diagnostic_is_deterministic_and_isolated():
    x = torch.linspace(0.0, 1.0, 8)
    y = x.clone()
    y_img = np.linspace(0.0, 1.0, 8)
    t_img = np.linspace(0.0, 0.3, 8)
    physics = {
        "a": 0.0,
        "b": 1.0,
        "c": 0.0,
        "d": 1.0,
        "interface_x": 0.5,
        "k_left": 2.0,
        "k_right": 1.0,
        "rho_left": 1.0,
        "rho_right": 1.0,
        "cp_left": 1.0,
        "cp_right": 1.0,
        "T_right": 300.0,
    }
    model = _PiecewiseSteadyModel()
    model.train()
    before_params = [parameter.detach().clone() for parameter in model.parameters()]
    training_rngs = HybridForcingRNGs.create(19, torch.device("cpu"))
    before_rng = training_rngs.state_dict()
    kwargs = dict(
        residual_method="hybrid", physics=physics, interface_face=3,
        exclusion=(float(0.5 * (x[2] + x[3])), float(0.5 * (x[4] + x[5]))),
        x_grid=x, y_grid=y, y_img=y_img, t_img=t_img, mu=300.0, sigma=100.0,
        q_ref=300.0, t_ref=0.3, t_final=0.3, t_ramp=0.0, dt=0.05,
        n_steps=6, intervals_per_sim=1, stratified=False, n_bins=2,
        n_r=4, n_bc=4, n_ic=4, chunk_r=0, seed=1234,
        weights={
            "interior": 1.0, "interface": 1.0, "left_neumann": 1.0,
            "topbot_adiabatic": 1.0, "ic": 1.0,
        },
        energy_cfg={"enabled": False},
        gradnorm_multipliers=None,
    )
    first = _forcing_diagnostic_gradient_metrics(model, **kwargs)
    second = _forcing_diagnostic_gradient_metrics(model, **kwargs)
    assert model.training
    for before, parameter in zip(before_params, model.parameters()):
        assert torch.equal(before, parameter.detach())
    _assert_rng_state_equal(before_rng, training_rngs.state_dict())
    for key, value in first.items():
        assert math.isfinite(value)
        assert second[key] == pytest.approx(value)


@pytest.mark.slow
def test_hybrid_objective_recovers_fixed_300_solution_without_saved_ic():
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, 8)
    y = x.clone()
    heat_flux, resistance = 0.0, 0.2
    mu, sigma = 300.0, 100.0
    physical = np.full((8, 8), 300.0, dtype=np.float32)
    params = [{
        "R_c": resistance,
        "interface_x": 0.5,
        "ic_family": "uniform_2d",
        "ic_params": {},
        "temporal_family": "pulse_train",
        "temporal_params": {
            "Np": 1, "A_list": [heat_flux], "t_list": [0.0], "dt_list": [0.3],
        },
        "spatial_family": "uniform",
        "spatial_params": {},
    }]
    physics = {
        "a": 0.0,
        "b": 1.0,
        "c": 0.0,
        "d": 1.0,
        "interface_x": 0.5,
        "k_left": 2.0,
        "k_right": 1.0,
        "rho_left": 1.0,
        "rho_right": 1.0,
        "cp_left": 1.0,
        "cp_right": 1.0,
        "T_right": 300.0,
    }
    model = _PiecewiseSteadyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    rngs = HybridForcingRNGs.create(3, torch.device("cpu"))
    y_img = np.linspace(0.0, 1.0, 8)
    t_img = np.linspace(0.0, 0.3, 8)
    exclusion = (
        float(0.5 * (x[2] + x[3])),
        float(0.5 * (x[4] + x[5])),
    )
    for _ in range(301):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = _forcing_layered_batch_losses(
            model, params, residual_method="hybrid", physics=physics,
            interface_face=3, exclusion=exclusion, x_grid=x, y_grid=y,
            y_img=y_img, t_img=t_img, mu=mu, sigma=sigma,
            q_ref=300.0, t_ref=0.3, t_final=0.3, t_ramp=0.0,
            dt=0.05, n_steps=6, intervals_per_sim=1,
            stratified=False, n_bins=2, n_r=16, n_bc=16, n_ic=64,
            chunk_r=0, rngs=rngs,
        )
        objective = sum(
            losses[name] for name in (
                "interior", "interface", "left_neumann",
                "topbot_adiabatic", "ic",
            )
        )
        objective.backward()
        optimizer.step()

    gx, gy = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack((gx.flatten(), gy.flatten()), dim=-1).unsqueeze(0)
    prediction = model.decode(
        model.encode(torch.zeros(1, 3, 8, 8), None, None),
        coords, torch.full((1, 64, 1), 0.15),
    ).detach().view(8, 8)
    target = torch.from_numpy((physical - mu) / sigma)
    assert float((prediction - target).square().mean().sqrt()) < 0.04
    assert abs(
        float((prediction[3] - prediction[4]).mean())
        - float((target[3] - target[4]).mean())
    ) < 0.03


def _write_forcing_dataset(path, num_sims=8, nt=4, nx=8, ny=8):
    rng = np.random.default_rng(11)
    trajectory = 300.0 + rng.normal(size=(num_sims, nt, nx, ny)).astype(np.float32)
    trajectory[:, 0] = 300.0
    trajectory[:, :, -1, :] = 300.0
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    time = np.linspace(0.0, 0.3, nt, dtype=np.float32)
    np.save(path / "trajectories.npy", trajectory)
    np.save(path / "x_grid.npy", x)
    np.save(path / "y_grid.npy", y)
    np.save(path / "t_grid.npy", time)
    np.save(path / "dt.npy", np.array(0.1))
    records = []
    for index in range(num_sims):
        records.append({
            "R_c": 0.1 + 0.1 * index,
            "T0": trajectory[index, 0].copy(),
            "ic_family": "uniform_2d",
            "ic_params": {"T0_offset": 0.0},
            "temporal_family": "sin",
            "temporal_params": {
                "A": 100.0 + index, "f": 2.0, "t_on": 0.0, "t_off": 0.2,
                "phase": 0.0, "tukey_alpha": 0.5,
            },
            "spatial_family": "uniform",
            "spatial_params": {},
        })
    np.save(path / "sim_params.npy", np.asarray(records, dtype=object))


def _forcing_config(path, residual_method):
    return {
        "benchmark": {"name": "forcing", "representation": "temporal_encoder"},
        "data": {
            "trajectories.npy": str(path / "trajectories.npy"),
            "x_grid_path": str(path / "x_grid.npy"),
            "y_grid_path": str(path / "y_grid.npy"),
            "t_grid_path": str(path / "t_grid.npy"),
        },
        "model": {
            "cvit": {
                "out_dim": 1, "emb_dim": 8, "patch_size": 4,
                "depth_enc": 1, "depth_dec": 1, "num_heads": 2,
                "mlp_ratio": 2.0, "fourier_freq": 1.0,
                "activation": "gelu", "hard_right_dirichlet": True,
                "hard_right_dirichlet_t_right": 300.0,
            },
            "interface_cvit": {
                "spatial_in_ch": 3, "forcing_in_ch": 1,
                "forcing_patch_size": 4, "n_param_scalars": 2,
                "num_param_tokens": 1, "param_hidden": 8,
            },
        },
        "training": {
            "device": "cpu", "epochs": 1, "validate_every": 1,
            "learning_rate": 1e-3, "weight_decay": 0.0,
            "optimizer": "Adam", "grad_clip": 1.0,
            "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.9},
            "gradnorm": {"enabled": False},
            "pino": {
                "residual_method": residual_method,
                "lambda_r": 1.0, "lambda_interface": 1.0,
                "lambda_ic": 1.0, "lambda_bc": 1.0,
                "lambda_bc_left": 1.0, "lambda_data": 0.0,
                "n_r": 4, "n_ic": 4, "n_bc": 4, "sim_batch": 2,
                "chunk_r": 0, "collocation_source": "online",
                "dt": 0.1, "intervals_per_sim": 1,
                "stratified_time_sampling": False, "causal_num_bins": 2,
                "causal": {"enabled": False},
                "region_standardize": {"enabled": False},
                "finite_volume": {
                    "interface_residual_weighting": {"enabled": True}
                },
                "forcing": {
                    "ny_img": 8, "nt_img": 8, "a_ref": 300.0,
                    "ramp_seconds": 0.01,
                },
            },
        },
    }


@pytest.mark.parametrize("residual_method", ["finite_volume", "hybrid"])
def test_forcing_interface_runner_smoke(tmp_path, residual_method):
    _write_forcing_dataset(tmp_path)
    run_dir = tmp_path / residual_method
    summary = run_one_seed_forcing_interface_pino(
        _forcing_config(tmp_path, residual_method), seed=2, run_dir=run_dir
    )
    assert summary["residual_method"] == residual_method
    with (run_dir / "train_metrics.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    for name in (
        "loss", "loss_interior", "loss_interface", "loss_left_neumann",
        "loss_top", "loss_bottom", "loss_ic", "val_gnrmse",
        "interface_flux_mismatch", "interface_contact_rmse_K",
        "interface_energy_rms", "gnrmse_lead_short", "gnrmse_lead_mid",
        "gnrmse_lead_long",
    ):
        assert math.isfinite(float(row[name])), name
    checkpoint = torch.load(
        run_dir / "cvit_best_global.pt", map_location="cpu", weights_only=False
    )
    manifest = checkpoint["physics_manifest"]
    assert manifest["residual_method"] == residual_method
    assert manifest["weights"] == {
        "interior": 1.0, "interface": 1.0, "left_neumann": 1.0,
        "topbot_adiabatic": 1.0, "ic": 1.0,
    }
    assert manifest["t_ref"] == pytest.approx(0.3)
    assert manifest["q_ref"] == pytest.approx(300.0)
    assert manifest["mu_global"] == pytest.approx(checkpoint["mu_global"])
    assert manifest["sigma_global"] == pytest.approx(checkpoint["sigma_global"])
    assert set(manifest["rng_state"]) == {
        "forcing_selection", "contact_resistance", "interval_selection",
        "bulk_collocation", "boundary_collocation", "ic_collocation",
        "validation",
    }
    assert manifest["training_problem_sampling"] == {
        "forcing": "online_all_families",
        "contact_resistance": "online_uniform",
        "R_c_range": [0.05, 1.0],
        "initial_condition_K": 300.0,
    }
    loaded_model, loaded = load_interface_cvit_checkpoint(
        run_dir / "cvit_best_global.pt", device=torch.device("cpu")
    )
    assert isinstance(loaded_model, InterfaceCViT)
    assert loaded["physics_manifest"]["residual_method"] == residual_method
    assert summary["gpu_hours"] >= 0.0
    assert math.isfinite(summary["best_validation_metrics"]["val_gnrmse"])
    with pytest.raises(FileExistsError, match="fresh experiment"):
        run_one_seed_forcing_interface_pino(
            _forcing_config(tmp_path, residual_method), seed=2, run_dir=run_dir
        )


def test_forcing_runner_fixed_probe_diagnostics_are_written(tmp_path):
    _write_forcing_dataset(tmp_path, num_sims=8, nt=4, nx=8, ny=8)
    config = _forcing_config(tmp_path, "hybrid")
    config["training"]["pino"]["diagnostics"] = {
        "enabled": True,
        "every_updates": 1,
        "gradient_n_r": 4,
        "gradient_n_bc": 4,
        "gradient_n_ic": 4,
        "probe_time_samples": 2,
        "probe_x_samples": 3,
        "probe_y_samples": 3,
    }
    run_dir = tmp_path / "diagnostics"
    run_one_seed_forcing_interface_pino(config, seed=4, run_dir=run_dir)
    assert (run_dir / "cvit_init.pt").exists()
    with (run_dir / "diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["completed_updates"]) for row in rows] == [0, 1]
    for name in (
        "drift_from_init_K", "departure_from_300_K", "field_rmse_K",
        "forcing_sensitivity_K", "forcing_response_ratio",
        "rc_sensitivity_K", "rc_response_ratio", "pred_jump_rms_K",
        "true_jump_rms_K", "node_jump_rmse_K",
        "probe_loss_energy_left", "probe_loss_energy_right",
        "probe_loss_energy_global", "probe_loss_energy",
        "grad_norm_discriminating", "grad_norm_stiff_physics",
        "grad_norm_homogeneous", "grad_cos_discriminating_physics",
        "effective_gradient_share_discriminating",
    ):
        assert math.isfinite(float(rows[0][name])), name
    assert float(rows[0]["drift_from_init_K"]) == pytest.approx(0.0, abs=1e-6)
    assert float(rows[1]["drift_from_init_K"]) >= 0.0


def test_forcing_runner_two_group_gradnorm_and_energy_are_logged(tmp_path):
    _write_forcing_dataset(tmp_path, num_sims=8, nt=4, nx=8, ny=8)
    config = _forcing_config(tmp_path, "hybrid")
    config["training"]["gradnorm"] = {
        "enabled": True, "alpha_w": 1.0, "update_every": 1,
        "eps": 1e-8, "w_min": 0.1, "w_max": 10.0, "floor": {},
    }
    config["training"]["pino"]["energy"] = {
        "enabled": True, "lambda": 1.0, "scale_floor": 1.0,
    }
    config["training"]["pino"]["diagnostics"] = {
        "enabled": True,
        "every_updates": 1,
        "gradient_n_r": 4,
        "gradient_n_bc": 4,
        "gradient_n_ic": 4,
        "probe_time_samples": 2,
        "probe_x_samples": 3,
        "probe_y_samples": 3,
        "probe_query_chunk": 16,
    }
    run_dir = tmp_path / "energy_gradnorm"
    run_one_seed_forcing_interface_pino(config, seed=5, run_dir=run_dir)
    with (run_dir / "train_metrics.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    for name in (
        "loss_energy_left", "loss_energy_right", "loss_energy_global",
        "loss_energy", "loss_group_discriminating",
        "loss_group_stiff_physics", "loss_group_homogeneous",
        "gradnorm_multiplier_discriminating",
        "gradnorm_multiplier_stiff_physics",
    ):
        assert math.isfinite(float(row[name])), name
    with (run_dir / "diagnostics.csv").open(newline="") as handle:
        diag0 = next(csv.DictReader(handle))
    for name in (
        "grad_norm_discriminating_weighted",
        "grad_norm_stiff_physics_weighted",
        "grad_ratio_discriminating_to_physics_weighted",
        "grad_projection_on_discriminating_weighted",
        "grad_projection_on_physics_weighted",
        "probe_loss_energy_left", "probe_loss_energy",
        "grad_norm_discriminating_decoder",
        "grad_norm_stiff_physics_forcing_encoder",
        "grad_norm_discriminating_param_encoder",
    ):
        assert math.isfinite(float(diag0[name])), name
    checkpoint = torch.load(
        run_dir / "cvit_last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["gradnorm_state"]["term_names"] == [
        "discriminating", "stiff_physics",
    ]
    assert checkpoint["physics_manifest"]["energy"]["enabled"] is True


def test_forcing_runner_can_pin_fixed_probe_case(tmp_path):
    _write_forcing_dataset(tmp_path, num_sims=8, nt=4, nx=8, ny=8)
    config = _forcing_config(tmp_path, "hybrid")
    config["training"]["pino"]["forcing"]["fixed_probe_case"] = "pulse_low"
    run_dir = tmp_path / "fixed_probe"
    run_one_seed_forcing_interface_pino(config, seed=6, run_dir=run_dir)
    checkpoint = torch.load(
        run_dir / "cvit_last.pt", map_location="cpu", weights_only=False
    )
    sampling = checkpoint["physics_manifest"]["training_problem_sampling"]
    assert sampling["forcing"] == "fixed_probe:pulse_low"
    assert sampling["contact_resistance"] == "fixed_probe"


def test_compositional_fixed_probe_cases_cross_family_and_resistance():
    pulse_mid = train_pino_module._forcing_fixed_probe_case("pulse_mid")
    smooth_low = train_pino_module._forcing_fixed_probe_case("smooth_low")
    smooth_high = train_pino_module._forcing_fixed_probe_case("smooth_high")
    assert pulse_mid["temporal_family"] == "pulse_train"
    assert pulse_mid["R_c"] == pytest.approx(0.525)
    assert smooth_low["temporal_family"] == "sin"
    assert smooth_low["R_c"] == pytest.approx(0.05)
    assert smooth_high["temporal_family"] == "sin"
    assert smooth_high["R_c"] == pytest.approx(1.0)


def test_forcing_reformulation_runner_logs_constraints_and_checkpoints_state(tmp_path):
    _write_forcing_dataset(tmp_path, num_sims=8, nt=4, nx=8, ny=8)
    config = _forcing_config(tmp_path, "hybrid")
    config["training"]["pino"]["forcing"]["fixed_probe_cases"] = [
        "pulse_low", "pulse_high",
    ]
    config["training"]["pino"]["reformulation"] = {
        "enabled": True, "x_blocks_per_layer": 2, "y_blocks": 2,
        "scale_floor": 1.0, "jump_floor_K": 1.0,
        "tolerance_margin": 10.0, "tolerance_min": 1.0e-8,
        "rho": 1.0, "ema_decay": 0.0, "dual_every": 1,
        "multiplier_cap": 1000.0, "dual_enabled": True,
        "causal_enabled": True, "stage_fractions": [0.5, 1.0],
        "stage_updates": [0, 1],
    }
    run_dir = tmp_path / "reformulation"
    run_one_seed_forcing_interface_pino(config, seed=7, run_dir=run_dir)
    with (run_dir / "train_metrics.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["causal_fraction"]) == pytest.approx(0.5)
    for name in train_pino_module._FORCING_CONSTRAINT_NAMES:
        assert math.isfinite(float(row[f"constraint_rms_{name}"]))
        assert math.isfinite(float(row[f"constraint_violation_{name}"]))
        assert math.isfinite(float(row[f"constraint_multiplier_{name}"]))
    checkpoint = torch.load(
        run_dir / "cvit_last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["constraint_state"] is not None
    assert checkpoint["hybrid_rng_state"] is not None
    assert checkpoint["physics_manifest"]["reformulation"]["measured_floors"][
        "left_flux"
    ] == 0.0


def test_forcing_supervised_metrics_distinguish_exact_jump_from_flat_field():
    truth = np.full((3, 4, 2), 300.0)
    truth[1:, 0] += 8.0
    truth[1:, 1] += 6.0
    truth[1:, 2] += 1.0
    initialization = np.full_like(truth, 300.0)

    exact = _forcing_supervised_metrics(truth, truth, initialization, 1)
    assert exact["field_rmse_K"] == pytest.approx(0.0)
    assert exact["node_jump_rmse_K"] == pytest.approx(0.0)
    assert exact["jump_response_ratio"] == pytest.approx(1.0)
    assert exact["temporal_response_ratio"] == pytest.approx(1.0)

    flat = _forcing_supervised_metrics(initialization, truth, initialization, 1)
    assert flat["field_signal_rel_l2"] == pytest.approx(1.0)
    assert flat["pred_jump_rms_K"] == pytest.approx(0.0)
    assert flat["node_jump_rmse_K"] == pytest.approx(flat["true_jump_rms_K"])
    assert flat["jump_response_ratio"] == pytest.approx(0.0)


def test_forcing_runner_rejects_dataset_solver_dt_conflict(tmp_path):
    _write_forcing_dataset(tmp_path)
    config = _forcing_config(tmp_path, "finite_volume")
    config["training"]["pino"]["dt"] = 0.05
    with pytest.raises(ValueError, match="conflicts with dataset dt.npy"):
        run_one_seed_forcing_interface_pino(
            config, seed=2, run_dir=tmp_path / "dt_conflict"
        )


def test_forcing_runner_never_reads_saved_training_problem_records(tmp_path):
    _write_forcing_dataset(tmp_path)
    records = np.load(tmp_path / "sim_params.npy", allow_pickle=True)
    train_ids, _, _ = split_sim_ids(len(records), seed=0)
    for sim_id in train_ids:
        records[int(sim_id)] = {"sentinel": "must_not_be_read"}
    np.save(tmp_path / "sim_params.npy", records)
    summary = run_one_seed_forcing_interface_pino(
        _forcing_config(tmp_path, "hybrid"),
        seed=2,
        run_dir=tmp_path / "online_only",
    )
    assert math.isfinite(summary["best_val_gnrmse"])


def test_forcing_benchmark_dispatches_to_layered_runner(tmp_path, monkeypatch):
    calls = []

    def fake_runner(config, seed, run_dir):
        calls.append((seed, run_dir))
        return {"seed": seed}

    monkeypatch.setattr(
        train_pino_module, "run_one_seed_forcing_interface_pino", fake_runner
    )
    config = {"benchmark": {"name": "forcing"}}
    result = run_config_seeds_pino(config, tmp_path, [2, 7])
    assert calls == [(2, tmp_path / "seed2"), (7, tmp_path / "seed7")]
    assert result["seeds"] == {"2": {"seed": 2}, "7": {"seed": 7}}


def test_forcing_constraint_dead_band_is_shared_by_primal_and_dual():
    tolerances = {
        name: 0.25 for name in train_pino_module._FORCING_CONSTRAINT_NAMES
    }
    controller = ForcingConstraintController(
        tolerances, rho=2.0, ema_decay=0.0, dual_every=1,
    )
    feasible = {
        name: torch.full((4,), 0.2, requires_grad=True)
        for name in tolerances
    }
    _, violation = controller.values(feasible)
    assert all(float(value) == 0.0 for value in violation.values())
    assert float(controller.primal(violation)) == 0.0
    controller.update(violation, completed=1)
    assert all(value == 0.0 for value in controller.multipliers.values())

    violated = {
        name: torch.full((4,), 0.5, requires_grad=True)
        for name in tolerances
    }
    _, violation = controller.values(violated)
    loss = controller.primal(violation)
    loss.backward()
    assert all(value.grad is not None for value in violated.values())
    controller.update(violation, completed=2)
    assert all(value == pytest.approx(0.5) for value in controller.multipliers.values())


def test_forcing_constraint_state_roundtrip_preserves_dual_history():
    tolerances = {
        name: 0.1 for name in train_pino_module._FORCING_CONSTRAINT_NAMES
    }
    source = ForcingConstraintController(tolerances, dual_every=1)
    residuals = {name: torch.ones(3) for name in tolerances}
    _, violation = source.values(residuals)
    source.update(violation, completed=1)
    restored = ForcingConstraintController(tolerances, dual_every=1)
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
    restored.reset_stage_history()
    assert restored.multipliers == source.multipliers
    assert not any(restored.ema_initialized.values())


def test_causal_interval_sampler_never_exceeds_active_prefix():
    sample = _sample_lattice_intervals(
        np.random.default_rng(4), batch_size=4, intervals_per_sim=3,
        n_steps=60, n_bins=6, stratified=True, max_start_step=5,
    )
    assert int(sample["start_idx"].max()) <= 5
    assert set(sample["start_idx"].tolist()) == set(range(6))
