import csv
import math

import numpy as np
import pytest
import torch

from data.dataset import split_sim_ids
from src.operators.losses import region_balanced_fv_rate_loss
import src.operators.train_pino as train_pino_module
from src.operators.train_pino import (
    load_interface_cvit_checkpoint,
    run_config_seeds_pino,
    run_one_seed_forcing_interface_pino,
)
from src.operators.cvit import InterfaceCViT
from src.operators.train_pino import (
    HybridForcingRNGs,
    _decode_derivatives,
    _forcing_layered_batch_losses,
    _sample_online_layered_forcing_params,
    _sample_hybrid_bulk,
)
from src.physics.fv_residual import (
    FullBCData,
    build_cn_geom_per_interface,
    full_bc_cn_residual,
    full_bc_cn_residual_rate,
    interface_residual_rate,
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
