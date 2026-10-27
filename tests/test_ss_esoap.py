import copy
from functools import reduce
import io
from pathlib import Path

import pytest
import torch

from src.operators.ss_esoap import SSESOAP, _project, _secant_energy
from src.operators.train import build_optimizer, load_config


@pytest.mark.parametrize('shape', [(3, 4), (2, 3, 2), (2, 2, 2, 2, 2)])
def test_inverse_secant_metric_matches_dense_kronecker(shape):
    torch.manual_seed(91)
    s = torch.randn(shape, dtype=torch.float64)
    factors = []
    for size in shape:
        a = torch.randn(size, size, dtype=s.dtype)
        factors.append(a @ a.T + .1 * torch.eye(size, dtype=s.dtype))
    metric = reduce(torch.kron, [torch.linalg.inv(a + .03 * torch.eye(a.shape[0], dtype=s.dtype)).contiguous() for a in factors])
    expected = s.flatten() @ metric @ s.flatten()
    torch.testing.assert_close(_secant_energy(s, factors, .03), expected)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_observed_near_singular_gram_matches_float64_inverse(device):
    covariance = torch.load(Path(__file__).parent / 'fixtures/ss_esoap_cholesky_failure.pt',
                            weights_only=True).to(device)
    original = covariance.clone()
    displacement = torch.linspace(-.02, .03, covariance.shape[0], device=device)
    precise = covariance.double() + 1e-8 * torch.eye(covariance.shape[0], device=device, dtype=torch.float64)
    assert torch.linalg.eigvalsh(precise).min() > 0
    expected = displacement.double() @ torch.linalg.solve(precise, displacement.double())
    actual = _secant_energy(displacement, [covariance], 1e-8)
    assert torch.isfinite(actual) and actual > 0
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(covariance, original, rtol=0, atol=0)


def test_negative_gram_roundoff_uses_existing_epsilon_spectral_floor():
    covariance = torch.diag(torch.tensor([.5, -2e-8]))
    displacement = torch.tensor([.003, .004])
    expected = .003 ** 2 / (.5 + 1e-8) + .004 ** 2 / 1e-8
    actual = _secant_energy(displacement, [covariance], 1e-8)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=2e-6, atol=1e-6)


def test_paper_update_uses_raw_gradient_secant_and_inverse_square_root_tau():
    p = torch.nn.Parameter(torch.tensor([[.1, .2], [.3, .5]], dtype=torch.float64))
    opt = SSESOAP([p], lr=.003, betas=(.9, .95), eps=.01, trigger_threshold=1)
    p.grad = p.detach().clone()
    opt.step()
    before = p.detach().clone()
    old = copy.deepcopy(opt.state[p])
    g = .2 * (before - old['previous_parameter']) + old['previous_gradient']
    s = before - old['previous_parameter']
    left = .95 * old['covariances'][0] + .05 * g @ g.T
    right = .95 * old['covariances'][1] + .05 * g.T @ g
    metric = torch.kron(torch.linalg.inv(left + .01 * torch.eye(2)).contiguous(),
                        torch.linalg.inv(right + .01 * torch.eye(2)).contiguous())
    energy = s.flatten() @ metric @ s.flatten()
    tau = (((g - old['previous_gradient']) * s).sum() / energy).clamp(.001, 1)
    assert tau < 1
    m = (.9 * old['momentum'] + .1 * g) / (1 - .9 ** 2)
    v = (.95 * old['variance'] + .05 * g.square()) / (1 - .95 ** 2)
    expected = before - .003 * tau.rsqrt() * m / (v.sqrt() + .01)
    p.grad = g.clone()
    opt.step()
    torch.testing.assert_close(p, expected, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(opt.state[p]['previous_gradient'], g)
    assert opt.state[p]['step'] == 2


def test_trigger_rotates_momentum_and_only_downscales_variance():
    p = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))
    opt = SSESOAP([p], betas=(.8, .9), trigger_threshold=.2)
    state = opt.state[p]
    g = torch.ones_like(p)
    opt._initialize(g, state, opt.param_groups[0])
    state['step'] = 1
    state['momentum'] = torch.arange(9, dtype=p.dtype).reshape_as(p)
    state['variance'] = torch.arange(1, 10, dtype=p.dtype).reshape_as(p)
    old = copy.deepcopy(state)
    opt._update_basis(g, state, opt.param_groups[0])
    assert state['basis_updates'] == 1
    assert state['gamma'] == .25
    torch.testing.assert_close(_project(state['momentum'], state['bases'], inverse=True), old['momentum'])
    torch.testing.assert_close(state['variance'], old['variance'] * .25)


@pytest.mark.parametrize('shape', [(), (4,), (3, 1, 1), (2, 3), (2, 2, 2, 2, 2)])
def test_first_call_updates_and_zero_gradients_are_finite(shape):
    p = torch.nn.Parameter(torch.ones(shape, dtype=torch.float64))
    opt = SSESOAP([p])
    p.grad = torch.full_like(p, .01)
    opt.step()
    assert not torch.equal(p, torch.ones_like(p))
    for _ in range(3):
        p.grad.zero_()
        opt.step()
    assert torch.isfinite(p).all()
    assert opt.state[p]['step'] == 4


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_checkpoint_resume_includes_secant_and_adaptive_state(device):
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(3, 4, 2, device=device))
    opt = SSESOAP([p])
    for _ in range(4):
        p.grad = p.detach() * .01
        opt.step()
    buffer = io.BytesIO()
    torch.save(opt.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.nn.Parameter(p.detach().clone())
    other = SSESOAP([restored])
    other.load_state_dict(torch.load(buffer, weights_only=False))
    for index in range(6):
        g = torch.randn_like(p) * .01 if index % 2 else p.detach() * .01
        for param, optimizer in [(p, opt), (restored, other)]:
            param.grad = g.clone()
            optimizer.step()
        torch.testing.assert_close(p, restored, rtol=0, atol=0)
    for key in ('step', 'basis_updates', 'scaling_updates', 'tau_floor_hits', 'gamma'):
        assert opt.state[p][key] == other.state[restored][key]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_full_spectral_tensor_cuda():
    p = torch.nn.Parameter(torch.randn(64, 64, 16, 16, 2, device='cuda') * .01)
    opt = SSESOAP([p])
    for _ in range(3):
        p.grad = p.detach() * .001
        opt.step()
    state = opt.state[p]
    assert state['step'] == 3
    assert len(state['bases']) == 5
    assert torch.isfinite(p).all()
    assert sum(t.numel() for t in state['bases']) < p.numel()


@pytest.mark.parametrize('kwargs', [dict(lr=-1), dict(betas=(1, .9)), dict(eps=0), dict(check_interval=0), dict(tau_min=0), dict(trigger_threshold=2)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SSESOAP([torch.nn.Parameter(torch.ones(2, 3))], **kwargs)


def test_shared_builder_defaults():
    cfg = load_config()
    cfg['training']['optimizer'] = 'SSESOAP'
    opt = build_optimizer(cfg, [torch.nn.Parameter(torch.ones(2, 3))])
    assert isinstance(opt, SSESOAP)
    assert opt.param_groups[0]['betas'] == (.9, .95)
