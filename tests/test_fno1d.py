import torch
import pytest

from src.operators.fno1d import SpectralConv1d, ConditioningMLP, ConditionalInstanceNorm1d, FNO1d


# ===================== SpectralConv1d =====================

class TestSpectralConv1d:
    def test_output_shape(self):
        layer = SpectralConv1d(12, 12, modes=4)
        x = torch.randn(2, 12, 101)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 12, 101)

    def test_output_shape_batch_1(self):
        layer = SpectralConv1d(12, 12, modes=4)
        x = torch.randn(1, 12, 101)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (1, 12, 101)

    def test_output_shape_small_grid(self):
        """Grid smaller than modes — min(modes, N_freq) guard is exercised."""
        layer = SpectralConv1d(4, 4, modes=16)
        x = torch.randn(2, 4, 8)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 4, 8)

    def test_output_finite(self):
        layer = SpectralConv1d(8, 8, modes=4)
        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            y = layer(x)
        assert torch.all(torch.isfinite(y))

    def test_different_in_out_channels(self):
        layer = SpectralConv1d(4, 8, modes=4)
        x = torch.randn(2, 4, 16)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 8, 16)

    def test_deterministic(self):
        layer = SpectralConv1d(4, 4, modes=4)
        layer.eval()
        x = torch.randn(1, 4, 16)
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x)
        assert torch.equal(y1, y2)


# ===================== ConditioningMLP =====================

class TestConditioningMLP:
    def test_output_shape(self):
        mlp = ConditioningMLP(cond_dim=7, hidden_dim=32, n_layers=4, width=8)
        cond = torch.randn(2, 7)
        out = mlp(cond)
        assert out.shape == (2, 4, 2, 8)  # (B, n_layers, 2, width)

    def test_identity_init_gamma_ones(self):
        """At initialization, γ should be ≈ 1."""
        mlp = ConditioningMLP(cond_dim=7, hidden_dim=32, n_layers=4, width=8)
        cond = torch.zeros(1, 7)  # zero input
        with torch.no_grad():
            out = mlp(cond)
        gamma = out[0, :, 0, :]  # (n_layers, width)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-5)

    def test_identity_init_beta_zeros(self):
        """At initialization, β should be ≈ 0."""
        mlp = ConditioningMLP(cond_dim=7, hidden_dim=32, n_layers=4, width=8)
        cond = torch.zeros(1, 7)
        with torch.no_grad():
            out = mlp(cond)
        beta = out[0, :, 1, :]  # (n_layers, width)
        assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-5)

    def test_gradients_flow(self):
        mlp = ConditioningMLP(cond_dim=7, hidden_dim=32, n_layers=4, width=8)
        cond = torch.randn(2, 7, requires_grad=True)
        out = mlp(cond)
        out.sum().backward()
        assert cond.grad is not None
        for name, p in mlp.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ===================== ConditionalInstanceNorm1d =====================

class TestConditionalInstanceNorm1d:
    def test_output_shape(self):
        cin = ConditionalInstanceNorm1d(8)
        x = torch.randn(2, 8, 64)
        gamma = torch.ones(2, 8)
        beta = torch.zeros(2, 8)
        out = cin(x, gamma, beta)
        assert out.shape == (2, 8, 64)

    def test_identity_with_gamma1_beta0(self):
        """With γ=1, β=0 the output should match plain InstanceNorm."""
        cin = ConditionalInstanceNorm1d(4)
        x = torch.randn(2, 4, 32)
        gamma = torch.ones(2, 4)
        beta = torch.zeros(2, 4)
        out = cin(x, gamma, beta)
        expected = cin.norm(x)
        assert torch.allclose(out, expected, atol=1e-6)


# ===================== FNO1d =====================

class TestFNO1d:
    def test_output_shape_standard(self, small_fno, synthetic_cond):
        x = torch.randn(4, 101, 2)
        cond = synthetic_cond
        with torch.no_grad():
            y = small_fno(x, cond)
        assert y.shape == (4, 101, 1)

    def test_output_shape_batch_1(self, small_fno):
        x = torch.randn(1, 101, 2)
        cond = torch.rand(1, 7)
        with torch.no_grad():
            y = small_fno(x, cond)
        assert y.shape == (1, 101, 1)

    def test_output_shape_batch_16(self, small_fno):
        x = torch.randn(16, 101, 2)
        cond = torch.rand(16, 7)
        with torch.no_grad():
            y = small_fno(x, cond)
        assert y.shape == (16, 101, 1)

    def test_output_finite(self, small_fno, synthetic_cond):
        x = torch.randn(4, 101, 2)
        with torch.no_grad():
            y = small_fno(x, synthetic_cond)
        assert torch.all(torch.isfinite(y))

    def test_padding_removed(self, small_fno):
        """Nx in output matches input despite internal padding."""
        x = torch.randn(1, 101, 2)
        cond = torch.rand(1, 7)
        with torch.no_grad():
            y = small_fno(x, cond)
        assert y.shape[1] == 101  # Nx preserved

    def test_parameter_count_positive(self, small_fno):
        n_params = sum(p.numel() for p in small_fno.parameters())
        assert n_params > 0

    def test_gradients_flow(self, small_fno, synthetic_cond):
        x = torch.randn(4, 101, 2)
        y = small_fno(x, synthetic_cond)
        loss = y.mean()
        loss.backward()
        for name, p in small_fno.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_through_conditioning(self, small_fno):
        """Verify gradients flow through the conditioning MLP."""
        x = torch.randn(2, 101, 2)
        cond = torch.randn(2, 7, requires_grad=True)
        y = small_fno(x, cond)
        y.sum().backward()
        assert cond.grad is not None

    def test_different_width(self):
        model = FNO1d(modes=2, width=32, in_channels=2, n_layers=2, cond_dim=7)
        x = torch.randn(1, 101, 2)
        cond = torch.rand(1, 7)
        with torch.no_grad():
            y = model(x, cond)
        assert y.shape == (1, 101, 1)

    def test_eval_mode_deterministic(self, small_fno):
        small_fno.eval()
        x = torch.randn(1, 101, 2)
        cond = torch.rand(1, 7)
        with torch.no_grad():
            y1 = small_fno(x, cond)
            y2 = small_fno(x, cond)
        assert torch.equal(y1, y2)

    def test_dropout_output_shape(self):
        model = FNO1d(modes=2, width=8, in_channels=2, n_layers=2, cond_dim=7, dropout=0.2)
        x = torch.randn(4, 101, 2)
        cond = torch.rand(4, 7)
        with torch.no_grad():
            y = model(x, cond)
        assert y.shape == (4, 101, 1)

    def test_dropout_train_vs_eval_differs(self):
        """Model with dropout should produce different outputs in train vs eval mode."""
        model = FNO1d(modes=2, width=8, in_channels=2, n_layers=2, cond_dim=7, dropout=0.5)
        x = torch.randn(4, 101, 2)
        cond = torch.rand(4, 7)

        model.eval()
        with torch.no_grad():
            y_eval = model(x, cond)

        model.train()
        torch.manual_seed(0)
        with torch.no_grad():
            y_train1 = model(x, cond)
        torch.manual_seed(1)
        with torch.no_grad():
            y_train2 = model(x, cond)

        # Training outputs should differ due to dropout randomness
        assert not torch.equal(y_train1, y_train2)

    def test_spectral_dropout_output_shape(self):
        model = FNO1d(modes=2, width=8, in_channels=2, n_layers=2, cond_dim=7, spectral_dropout=0.3)
        x = torch.randn(4, 101, 2)
        cond = torch.rand(4, 7)
        with torch.no_grad():
            y = model(x, cond)
        assert y.shape == (4, 101, 1)

    def test_combined_dropout_and_spectral_dropout(self):
        model = FNO1d(modes=2, width=8, in_channels=2, n_layers=2, cond_dim=7, dropout=0.2, spectral_dropout=0.2)
        x = torch.randn(4, 101, 2)
        cond = torch.rand(4, 7)
        with torch.no_grad():
            y = model(x, cond)
        assert y.shape == (4, 101, 1)
        assert torch.all(torch.isfinite(y))

    def test_zero_dropout_matches_default(self):
        """dropout=0.0 should behave identically to no dropout."""
        torch.manual_seed(42)
        model_no_drop = FNO1d(modes=2, width=8, in_channels=2, n_layers=2, cond_dim=7)
        torch.manual_seed(42)
        model_zero_drop = FNO1d(modes=2, width=8, in_channels=2, n_layers=2, cond_dim=7, dropout=0.0)

        model_no_drop.eval()
        model_zero_drop.eval()
        x = torch.randn(1, 101, 2)
        cond = torch.rand(1, 7)
        with torch.no_grad():
            y1 = model_no_drop(x, cond)
            y2 = model_zero_drop(x, cond)
        assert torch.equal(y1, y2)
