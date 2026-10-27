"""Independent SS-eSOAP implementation of arXiv:2608.29448v1, Algorithm 2.

This follows the paper rather than the materially different public SS_SOAP
prototype (al2wang/ss_esoap_nips, 2d02e10378c927ecd496d65d8c2d4e76045947ef).
For real N-D tensors, each mode has an unnormalized gradient Gram factor;
the secant metric is the product of their regularized inverses. This is a
local tensor extension, not a published FNO result. Scalar/vector parameters
use ordinary Adam. Singleton axes are omitted. Every call performs an update.
Secants use consecutive supplied gradients (including any external clipping).
With fresh minibatches these contain sampling noise, not just curvature.
"""

import math

import torch


def _project(value, bases, inverse=False):
    for axis, basis in enumerate(bases):
        matrix = basis if inverse else basis.T
        value = torch.tensordot(matrix, value, dims=([1], [axis])).movedim(0, axis)
    return value


def _secant_energy(displacement, covariances, eps):
    # Triangular whitening evaluates the actual current inverse metric even
    # when the adaptively retained eigenbasis is stale (Appendix B.3 assumes
    # exact eigenvectors). No flattened Kronecker matrix is constructed.
    whitened = displacement
    for axis, covariance in enumerate(covariances):
        # A 256-wide factor from the real run was SPD in float64 but failed
        # float32 Cholesky at condition number ~1e7. Keep the existing damping
        # and Gram state, but factor the small symmetric matrix in float64.
        precise = covariance.double()
        regularized = (precise + precise.T) * 0.5
        regularized.diagonal().add_(eps)
        factor, info = torch.linalg.cholesky_ex(regularized)
        # CUDA TRSM is extremely slow for the two-entry real/imaginary axis
        # with a million RHS columns. Invert only this small triangular factor
        # and apply it with GEMM; the whitening metric is unchanged.
        if info.item() == 0:
            inverse_factor = torch.linalg.solve_triangular(
                factor, torch.eye(factor.shape[0], device=factor.device, dtype=factor.dtype),
                upper=False)
        else:
            # A Gram EMA is PSD in exact arithmetic. Project away negative
            # roundoff eigenvalues if float32 accumulation already lost PSD;
            # eps remains the spectral floor, without a new damping setting.
            values, basis = torch.linalg.eigh(regularized)
            inverse_factor = values.clamp_min(eps).rsqrt()[:, None] * basis.T
        inverse_factor = inverse_factor.to(displacement.dtype)
        moved = whitened.movedim(axis, 0)
        whitened = (inverse_factor @ moved.reshape(moved.shape[0], -1)).reshape(
            moved.shape).movedim(0, axis)
    return whitened.double().square().sum()


class SSESOAP(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, check_interval=1, trigger_threshold=0.2,
                 warmup_steps=0, tau_min=1e-3, max_precond_dim=2048):
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps,
                        weight_decay=weight_decay, check_interval=check_interval,
                        trigger_threshold=trigger_threshold, warmup_steps=warmup_steps,
                        tau_min=tau_min, max_precond_dim=max_precond_dim)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for key in ('lr', 'eps', 'weight_decay', 'trigger_threshold', 'tau_min'):
                value = group[key]
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid SSESOAP {key}: {value}')
            if group['eps'] == 0 or not 0 < group['tau_min'] <= 1:
                raise ValueError('SSESOAP requires eps > 0 and tau_min in (0, 1]')
            if not 0 <= group['trigger_threshold'] <= 1:
                raise ValueError('SSESOAP trigger_threshold must be in [0, 1]')
            if len(group['betas']) != 2 or any(not 0 <= b < 1 for b in group['betas']):
                raise ValueError('SSESOAP betas must be in [0, 1)')
            for key in ('check_interval', 'max_precond_dim', 'warmup_steps'):
                if not isinstance(group[key], int) or group[key] < (0 if key == 'warmup_steps' else 1):
                    raise ValueError(f'Invalid SSESOAP {key}')
            for p in group['params']:
                if p.dtype not in (torch.float32, torch.float64):
                    raise ValueError('SSESOAP supports real float32/float64 parameters')
                shape = [n for n in p.shape if n > 1]
                if len(shape) > 1 and max(shape) > group['max_precond_dim']:
                    raise ValueError('SSESOAP tensor axis exceeds max_precond_dim')

    def _initialize(self, grad, state, group):
        state.update(step=0, basis_updates=0, scaling_updates=0, tau_floor_hits=0,
                     tau=grad.new_tensor(1.0), rho=grad.new_tensor(0.0), gamma=1.0,
                     momentum=torch.zeros_like(grad), variance=torch.zeros_like(grad),
                     covariances=[], bases=[])
        if grad.ndim > 1:
            for size in grad.shape:
                identity = torch.eye(size, dtype=grad.dtype, device=grad.device)
                state['covariances'].append(group['eps'] * identity)
                state['bases'].append(identity)

    def _update_basis(self, grad, state, group):
        beta2 = group['betas'][1]
        for axis, covariance in enumerate(state['covariances']):
            unfolding = grad.movedim(axis, 0).reshape(grad.shape[axis], -1)
            covariance.lerp_(unfolding @ unfolding.T, 1 - beta2)
        state['gamma'] = 1.0
        if (not state['bases'] or state['step'] <= group['warmup_steps']
                or state['step'] % group['check_interval']):
            return
        ratios = []
        for covariance, basis in zip(state['covariances'], state['bases']):
            rotated = basis.T @ covariance @ basis
            offdiag = rotated - torch.diag(rotated.diagonal())
            ratios.append(torch.linalg.vector_norm(offdiag) /
                          (torch.linalg.vector_norm(rotated) + group['eps']))
        rho = torch.stack(ratios).max()
        state['rho'] = rho
        if rho.item() <= group['trigger_threshold']:
            return
        physical_momentum = _project(state['momentum'], state['bases'], inverse=True)
        state['bases'] = [torch.linalg.eigh(covariance)[1].flip(1)
                          for covariance in state['covariances']]
        state['momentum'] = _project(physical_momentum, state['bases'])
        gamma = 0.25 if rho.item() > 0.8 else 0.5 if rho.item() > 0.5 else 0.75
        state['variance'].mul_(gamma)
        state['gamma'] = gamma
        state['basis_updates'] += 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError('SSESOAP does not support sparse gradients')
                shape = tuple(n for n in p.shape if n > 1) or (1,)
                grad = p.grad.reshape(shape)
                state = self.state[p]
                if not state:
                    self._initialize(grad, state, group)
                state['step'] += 1
                self._update_basis(grad, state, group)
                projected = _project(grad, state['bases'])
                beta1, beta2 = group['betas']
                state['momentum'].lerp_(projected, 1 - beta1)
                state['variance'].lerp_(projected.square(), 1 - beta2)
                tau = grad.new_tensor(1.0)
                if state['bases'] and 'previous_parameter' in state:
                    displacement = p.detach().reshape(shape) - state['previous_parameter']
                    delta_grad = grad - state['previous_gradient']
                    curvature = (delta_grad.double() * displacement.double()).sum()
                    if curvature.item() > 0:
                        energy = _secant_energy(displacement, state['covariances'], group['eps'])
                        if energy.item() > 0:
                            tau = (curvature / energy).clamp(group['tau_min'], 1).to(grad.dtype)
                state['tau'] = tau
                state['scaling_updates'] += int(tau.item() < 1)
                state['tau_floor_hits'] += int((tau <= group['tau_min']).item())
                if state['bases']:
                    state['previous_parameter'] = p.detach().reshape(shape).clone()
                    state['previous_gradient'] = grad.clone()
                momentum = state['momentum'] / (1 - beta1 ** state['step'])
                variance = state['variance'] / (1 - beta2 ** state['step'])
                direction = _project(tau.rsqrt() * momentum / (variance.sqrt() + group['eps']),
                                     state['bases'], inverse=True)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(direction.reshape_as(p), alpha=-group['lr'])
        return loss
