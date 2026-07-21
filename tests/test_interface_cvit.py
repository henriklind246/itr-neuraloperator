import inspect

import numpy as np
import pytest
import torch

from problems.interfaces import (
    build_cond_vector,
    normalize_interface_scalars,
    INTERFACE_X_RANGE,
    RC_RANGE,
)
from src.operators.cvit import (
    CViT,
    ForcingCViT,
    InterfaceCViT,
    moving_interface_jump_enrichment,
)
from src.operators.train_pino import build_cvit, load_interface_cvit_checkpoint

"""
Deterministic architecture / wiring tests for `InterfaceCViT`. These are exact,
seed-free assertions on how the three token streams are wired (spatial field /
forcing image / interface scalars) and on the function-value information
contract `(x_Gamma, R_c, q_L(y,t)) -> T` — NOT claims about
learned behavior. They pin:
  - the model consumes only `(u_spatial, forcing_image, params=[x_hat, Rc_hat])`,
    never the raw generator parameters;
  - forcing / parameter tokens depend deterministically on their inputs and are
    on the autograd path;
  - `x_hat`/`Rc_hat` share the linear `build_cond_vector` normalization;
  - the untouched `CViT`/`ForcingCViT` classes still construct and run.
"""

GRID = (20, 20)
FORCING_GRID = (16, 32)
EMB = 32


def _model(**kw):
    defaults = dict(
        spatial_in_ch=3, emb_dim=EMB, patch_size=10, grid_size=GRID,
        depth_enc=1, depth_dec=1, num_heads=4, t_final=0.2,
        forcing_patch_size=8, forcing_grid_size=FORCING_GRID,
        num_param_tokens=1,
    )
    defaults.update(kw)
    m = InterfaceCViT(**defaults)
    return m.eval()


def _inputs(B=2, Nq=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, 3, *GRID, generator=g)
    forcing_image = torch.randn(B, 1, *FORCING_GRID, generator=g)
    params = torch.rand(B, 2, generator=g)
    coords = torch.rand(B, Nq, 2, generator=g)
    t = torch.rand(B, Nq, 1, generator=g) * 0.2
    return u, forcing_image, params, coords, t


def test_model_smoke_shapes():
    m = _model()
    u, forcing_image, params, coords, t = _inputs()
    out = m(u, coords, t, forcing_image, params)
    assert out.shape == (2, 7, 1)
    lat = m.encode(u, forcing_image, params)
    assert lat.shape == (2, 4 + 8 + 1, EMB)


def test_moving_jump_enrichment_shifts_only_dynamic_left_side():
    smooth = torch.zeros(2, 4, 1)
    coords = torch.tensor([
        [[0.1, 0.2], [0.4, 0.2], [0.6, 0.2], [1.0, 0.2]],
        [[0.1, 0.2], [0.4, 0.2], [0.6, 0.2], [1.0, 0.2]],
    ])
    flux = torch.full_like(smooth, 2.0)
    enriched = moving_interface_jump_enrichment(
        smooth, coords, flux, torch.tensor([0.3, 0.7]), torch.tensor([3.0, 4.0])
    )
    assert torch.equal(enriched[0, :, 0], torch.tensor([6.0, 0.0, 0.0, 0.0]))
    assert torch.equal(enriched[1, :, 0], torch.tensor([8.0, 8.0, 8.0, 0.0]))


def test_jump_flux_head_is_zero_initialized_and_reaches_parameters():
    model = _model(jump_enrichment=True).train()
    u, forcing_image, params, coords, t = _inputs(B=2, Nq=9)
    latent = model.encode(u, forcing_image, params)
    flux = model.decode_interface_flux(latent, coords[..., 1:2], t)
    assert torch.equal(flux, torch.zeros_like(flux))
    loss = (flux - 1.0).square().mean()
    loss.backward()
    last = model.interface_flux_decoder.head.net[-1]
    assert float(last.weight.grad.abs().sum()) > 0.0


def test_state_param_flux_head_is_invariant_to_forcing_tokens():
    torch.manual_seed(4)
    model = _model(
        jump_enrichment=True, jump_flux_conditioning="state_param"
    ).eval()
    with torch.no_grad():
        model.interface_flux_decoder.head.net[-1].weight.normal_()
    u, forcing_image, params, coords, t = _inputs(B=2, Nq=9, seed=5)
    latent = model.encode(u, forcing_image, params)
    baseline = model.decode_interface_flux(latent, coords[..., 1:2], t)
    changed_forcing = latent.clone()
    start = model.num_spatial_tokens
    stop = start + model.num_forcing_tokens
    changed_forcing[:, start:stop] += 100.0
    forcing_result = model.decode_interface_flux(
        changed_forcing, coords[..., 1:2], t
    )
    assert torch.equal(baseline, forcing_result)
    changed_state = latent.clone()
    changed_state[:, :start] += torch.randn_like(changed_state[:, :start])
    state_result = model.decode_interface_flux(changed_state, coords[..., 1:2], t)
    assert not torch.allclose(baseline, state_result)


def test_rejects_unknown_jump_flux_conditioning():
    with pytest.raises(ValueError, match="jump_flux_conditioning"):
        _model(jump_flux_conditioning="forcing_only")


def test_left_energy_closure_has_no_learned_flux_decoder():
    model = _model(
        jump_enrichment=True, jump_flux_mode="left_energy_closure"
    )
    assert not hasattr(model, "interface_flux_decoder")
    u, forcing_image, params, coords, t = _inputs()
    latent = model.encode(u, forcing_image, params)
    with pytest.raises(RuntimeError, match="disabled"):
        model.decode_interface_flux(latent, coords[..., 1:2], t)


def test_two_sided_energy_closure_has_no_learned_flux_decoder():
    model = _model(
        jump_enrichment=True, jump_flux_mode="two_sided_energy_closure"
    )
    assert not hasattr(model, "interface_flux_decoder")


def test_conservative_storage_projection_has_no_learned_flux_decoder():
    model = _model(
        jump_enrichment=True, jump_flux_mode="conservative_storage_projection"
    )
    assert not hasattr(model, "interface_flux_decoder")


def test_rejects_unknown_jump_flux_mode():
    with pytest.raises(ValueError, match="jump_flux_mode"):
        _model(jump_flux_mode="explicit_guess")


def test_encode_signature_excludes_generator_params():
    """The function-value contract: `encode` takes only the spatial field, the
    forcing image, and the two interface scalars — there is no argument path
    for amplitude/frequency/phase."""
    sig = list(inspect.signature(InterfaceCViT.encode).parameters)
    assert sig == ["self", "u_spatial", "forcing_image", "params"]
    for bad in ("amplitude", "frequency", "phase", "temporal_params"):
        assert bad not in sig


def test_external_waveform_record_cannot_change_fixed_model_inputs():
    model = _model()
    u, forcing_image, params, coords, t = _inputs()
    external_record = {"A": 50.0, "f": 2.0, "phase": 0.0}
    first = model(u, coords, t, forcing_image, params)
    external_record.update({"A": 300.0, "f": 20.0, "phase": 1.5})
    second = model(u, coords, t, forcing_image, params)
    assert torch.equal(first, second)


def test_forcing_tokens_track_image():
    m = _model()
    _, forcing_image, _, _, _ = _inputs()
    zf_a = m.forcing_encoder(forcing_image)
    zf_b = m.forcing_encoder(forcing_image.clone())
    assert torch.equal(zf_a, zf_b)
    image2 = forcing_image.clone()
    image2[0] += 1.0
    zf_c = m.forcing_encoder(image2)
    assert not torch.allclose(zf_a[0], zf_c[0])  # sim 0 token changed
    assert torch.equal(zf_a[1], zf_c[1])         # sim 1 untouched


def test_param_token_tracks_scalars():
    m = _model()
    p = torch.rand(3, 2)
    zp = m.param_encoder(p)
    assert torch.equal(zp, m.param_encoder(p.clone()))
    p2 = p.clone()
    p2[0, 1] += 0.3  # change Rc_hat of sim 0
    zp2 = m.param_encoder(p2)
    assert not torch.allclose(zp[0], zp2[0])
    assert torch.equal(zp[1:], zp2[1:])


def test_gradients_reach_forcing_and_param_branches():
    m = _model().train()
    u, forcing_image, params, coords, t = _inputs()
    out = m(u, coords, t, forcing_image, params)
    out.sum().backward()
    fp = list(m.forcing_encoder.parameters())
    pp = list(m.param_encoder.parameters())
    assert all(p.grad is not None for p in fp)
    assert all(p.grad is not None for p in pp)
    assert m.modality.grad is not None


def test_double_backward_through_decode():
    """The PDE residual differentiates the decoder twice w.r.t. coords with
    create_graph=True; the explicit-attention decoder must survive it."""
    m = _model()
    u, forcing_image, params, _, _ = _inputs()
    lat = m.encode(u, forcing_image, params)
    B = u.shape[0]
    x = torch.rand(B, 5, 1, requires_grad=True)
    y = torch.rand(B, 5, 1, requires_grad=True)
    t = torch.rand(B, 5, 1, requires_grad=True) * 0.2
    T = m.decode(lat, torch.cat([x, y], dim=-1), t)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    assert T_xx.shape == (B, 5, 1)
    (T_xx ** 2).mean().backward()  # third backward into params must not raise


def test_hard_right_dirichlet_exact_at_x1():
    m = _model()
    u, forcing_image, params, coords, t = _inputs()
    lat = m.encode(u, forcing_image, params)
    coords1 = coords.clone()
    coords1[..., 0] = 1.0  # x = 1 wall
    out = m.decode(lat, coords1, t)
    assert float(out.abs().max()) == pytest.approx(0.0, abs=1e-6)  # t_right_tilde=0


def test_encode_once_decode_twice_matches_single_shot():
    m = _model()
    u, forcing_image, params, coords, t = _inputs()
    lat = m.encode(u, forcing_image, params)
    d = m.decode(lat, coords, t)
    f = m(u, coords, t, forcing_image, params)
    assert torch.allclose(d, f, atol=1e-6)


def test_batch_permutation_equivariance():
    m = _model()
    u, forcing_image, params, coords, t = _inputs(B=3)
    out = m(u, coords, t, forcing_image, params)
    perm = torch.tensor([2, 0, 1])
    out_p = m(
        u[perm], coords[perm], t[perm], forcing_image[perm], params[perm]
    )
    assert torch.allclose(out_p, out[perm], atol=1e-6)


def test_changing_one_sim_forcing_isolates_to_that_sim():
    m = _model()
    u, forcing_image, params, coords, t = _inputs(B=3)
    out = m(u, coords, t, forcing_image, params)
    image2 = forcing_image.clone()
    image2[1] += 2.0
    out2 = m(u, coords, t, image2, params)
    assert not torch.allclose(out[1], out2[1])           # sim 1 responds
    assert torch.allclose(out[0], out2[0], atol=1e-6)    # sims 0, 2 unchanged
    assert torch.allclose(out[2], out2[2], atol=1e-6)


@pytest.mark.parametrize(
    "ix,rc",
    [(0.2, 0.05), (0.5, 0.5), (0.8, 1.0), (0.35, 0.7)],
)
def test_scalar_normalization_matches_cond_vector(ix, rc):
    """`normalize_interface_scalars` reproduces the R_c / interface_x entries of
    `build_cond_vector` (indices 2, 3)."""
    xhat, rchat = normalize_interface_scalars(ix, rc)
    cv = build_cond_vector(0.0, 0.0, rc, ix)
    assert rchat == pytest.approx(cv[2], abs=1e-6)  # R_c_norm
    assert xhat == pytest.approx(cv[3], abs=1e-6)   # interface_x_norm
    assert 0.0 - 1e-6 <= xhat <= 1.0 + 1e-6
    assert 0.0 - 1e-6 <= rchat <= 1.0 + 1e-6


def test_normalize_interface_scalars_vectorized():
    ix = np.array([0.2, 0.5, 0.8], dtype=np.float64)  # over (0.2, 0.8)
    rc = np.array([0.05, 1.0], dtype=np.float64)       # endpoints of (0.05, 1.0)
    out_x = normalize_interface_scalars(ix, np.full_like(ix, 0.05))
    assert out_x.shape == (3, 2)
    np.testing.assert_allclose(out_x[:, 0], [0.0, 0.5, 1.0], atol=1e-6)
    out_rc = normalize_interface_scalars(np.full_like(rc, 0.5), rc)
    np.testing.assert_allclose(out_rc[:, 1], [0.0, 1.0], atol=1e-6)


def test_build_cvit_interfaces_variant():
    config = {
        "model": {
            "cvit": {
                "emb_dim": EMB, "patch_size": 10, "depth_enc": 1, "depth_dec": 1,
                "num_heads": 4, "mlp_ratio": 2.0,
            },
            "interface_cvit": {
                "spatial_in_ch": 3, "forcing_in_ch": 1,
                "forcing_patch_size": 8, "num_param_tokens": 1,
            },
        },
        "training": {"pino": {"forcing": {
            "ny_img": FORCING_GRID[0], "nt_img": FORCING_GRID[1], "a_ref": 300.0,
        }}},
    }
    m = build_cvit(config, mu=300.0, sigma=10.0, grid_size=GRID, t_final=0.2,
                   variant="interfaces")
    assert isinstance(m, InterfaceCViT)
    u, forcing_image, params, coords, t = _inputs()
    out = m(u, coords, t, forcing_image, params)
    assert out.shape == (2, 7, 1)


def test_production_token_contract_is_384_plus_100_plus_one():
    m = InterfaceCViT(
        spatial_in_ch=3, emb_dim=16, patch_size=10, grid_size=(100, 100),
        forcing_patch_size=8, forcing_grid_size=(96, 256), depth_enc=0,
        depth_dec=1, num_heads=4,
    )
    assert m.forcing_encoder.pos_emb.shape[1] == 384
    assert m.spatial_encoder.pos_emb.shape[1] == 100
    assert 384 + 100 + m.param_encoder.num_tokens == 485


def test_rejects_invalid_forcing_image_contract():
    with pytest.raises(ValueError, match="divisible"):
        _model(forcing_grid_size=(95, 256))
    m = _model()
    u, forcing_image, params, _, _ = _inputs()
    with pytest.raises(ValueError, match="forcing_image must have shape"):
        m.encode(u, forcing_image[..., :-1], params)
    with pytest.raises(ValueError, match="params must have shape"):
        m.encode(u, forcing_image, torch.zeros(u.shape[0], 3))


def test_rejects_legacy_waveform_configuration():
    config = {
        "model": {
            "cvit": {"emb_dim": EMB, "patch_size": 10},
            "interface_cvit": {"temporal_samples": 128},
        },
        "training": {"pino": {"forcing": {
            "ny_img": 16, "nt_img": 32, "a_ref": 300.0,
        }}},
    }
    with pytest.raises(ValueError, match="no longer supports waveform-token"):
        build_cvit(config, 300.0, 10.0, GRID, 0.2, variant="interfaces")


def test_legacy_waveform_state_fails_before_generic_state_mismatch():
    m = _model()
    with pytest.raises(RuntimeError, match="waveform_tokens"):
        m.load_state_dict({"forcing_encoder.lift.weight": torch.zeros(1)})


def test_checkpoint_round_trip_uses_saved_config_and_image_spec(tmp_path):
    config = {
        "model": {
            "cvit": {
                "emb_dim": EMB, "patch_size": 10, "depth_enc": 1,
                "depth_dec": 1, "num_heads": 4, "mlp_ratio": 2.0,
            },
            "interface_cvit": {
                "spatial_in_ch": 3, "forcing_in_ch": 1,
                "forcing_patch_size": 8, "num_param_tokens": 1,
            },
        },
        "training": {"pino": {"forcing": {
            "ny_img": 16, "nt_img": 32, "a_ref": 300.0,
        }}},
    }
    model = build_cvit(
        config, mu=300.0, sigma=10.0, grid_size=GRID, t_final=0.2,
        variant="interfaces",
    ).eval()
    spec = {
        "representation": "space_time_image", "version": 1,
        "forcing_schema_version": 1, "axis_order": "channel_y_time",
        "dtype": "float32", "ny_img": 16, "nt_img": 32,
        "patch_size": 8, "include_endpoints": True, "y_min": 0.0,
        "y_max": 1.0, "t_min": 0.0, "t_final": 0.2,
        "sign_convention": "positive_inward_left_flux",
        "normalization": "fixed_division", "a_ref": 300.0,
        "clipping": False,
        "ramp": {"type": "cubic_smoothstep", "version": 1, "duration": 0.01},
        "spatial_grid_size": list(GRID),
    }
    path = tmp_path / "interface.pt"
    torch.save({
        "model_state": model.state_dict(), "mu_global": 300.0,
        "sigma_global": 10.0, "config": config, "interface_forcing": spec,
    }, path)
    restored, checkpoint = load_interface_cvit_checkpoint(path)
    restored.eval()
    u, forcing_image, params, coords, t = _inputs()
    torch.testing.assert_close(
        restored(u, coords, t, forcing_image, params),
        model(u, coords, t, forcing_image, params),
        rtol=0.0, atol=0.0,
    )
    assert checkpoint["interface_forcing"] == spec

    legacy_path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}, "config": config}, legacy_path)
    with pytest.raises(RuntimeError, match="Legacy waveform-token checkpoints"):
        load_interface_cvit_checkpoint(legacy_path)


def test_untouched_cvit_forcingcvit_still_build():
    """`InterfaceCViT` is a separate class; the legacy paths are unchanged."""
    c = CViT(in_ch=1, emb_dim=EMB, patch_size=10, grid_size=GRID,
             depth_enc=1, depth_dec=1, num_heads=4)
    f = ForcingCViT(in_ch=1, emb_dim=EMB, patch_size=10, grid_size=GRID,
                    depth_enc=1, depth_dec=1, num_heads=4)
    assert hasattr(c, "encoder") and hasattr(c, "decoder")
    assert isinstance(f, CViT)  # ForcingCViT subclasses CViT
    assert not isinstance(InterfaceCViT(spatial_in_ch=3, emb_dim=EMB,
                                        patch_size=10, grid_size=GRID,
                                        depth_enc=1, depth_dec=1, num_heads=4), CViT)
