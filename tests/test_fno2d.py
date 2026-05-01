import torch

from src.operators.fno2d import ConditionalInstanceNorm2d, FNO2d


class TestConditionalInstanceNorm2d:
    def test_matches_unfused_formula_and_gradients(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=8)
        x = torch.randn(2, 8, 12, 12, requires_grad=True)
        gamma = torch.randn(2, 8, requires_grad=True)
        beta = torch.randn(2, 8, requires_grad=True)

        y_new = cin(x, gamma, beta, valid_shape=(10, 10))

        valid = x[..., :10, :10]
        var, mean = torch.var_mean(valid, dim=(-2, -1), unbiased=False, keepdim=True)
        y_old = (x - mean) * torch.rsqrt(var + cin.eps)
        y_old = gamma[:, :, None, None] * y_old + beta[:, :, None, None]

        assert torch.allclose(y_new, y_old, atol=1e-6)

        grads_new = torch.autograd.grad(y_new.sum(), (x, gamma, beta), retain_graph=True)
        grads_old = torch.autograd.grad(y_old.sum(), (x, gamma, beta), retain_graph=True)
        for grad_new, grad_old in zip(grads_new, grads_old):
            assert torch.allclose(grad_new, grad_old, atol=1e-5)

    def test_physical_region_has_instance_norm_statistics(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=3)
        x = torch.randn(2, 3, 4, 5)
        gamma = torch.ones(2, 3)
        beta = torch.zeros(2, 3)

        out = cin(x, gamma, beta, valid_shape=(4, 5))

        mean = out.mean(dim=(-2, -1))
        var = out.var(dim=(-2, -1), unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_padded_region_does_not_affect_physical_output(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=3)
        physical = torch.randn(2, 3, 4, 5)
        x_a = torch.zeros(2, 3, 6, 7)
        x_b = torch.full((2, 3, 6, 7), 1000.0)
        x_a[..., :4, :5] = physical
        x_b[..., :4, :5] = physical
        gamma = torch.ones(2, 3)
        beta = torch.zeros(2, 3)

        out_a = cin(x_a, gamma, beta, valid_shape=(4, 5))
        out_b = cin(x_b, gamma, beta, valid_shape=(4, 5))

        assert torch.allclose(out_a[..., :4, :5], out_b[..., :4, :5])

    def test_defaults_to_full_tensor_when_no_valid_shape_is_given(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=3)
        x = torch.randn(2, 3, 4, 5)
        gamma = torch.ones(2, 3)
        beta = torch.zeros(2, 3)

        out = cin(x, gamma, beta)

        mean = out.mean(dim=(-2, -1))
        var = out.var(dim=(-2, -1), unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


class TestFNO2d:
    def test_forward_returns_channels_last_output(self):
        model = FNO2d(modes1=2, modes2=2, width=8, in_channels=8, out_channels=1, n_layers=2, cond_dim=28)
        spatial = torch.randn(2, 11, 11, 8)
        cond = torch.randn(2, 28)

        out = model(spatial, cond)

        assert out.shape == (2, 11, 11, 1)
