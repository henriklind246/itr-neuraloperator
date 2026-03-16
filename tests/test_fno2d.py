import torch
import pytest

from src.operators.fno2d import SpectralConv2d, FNO2d


# ===================== SpectralConv2d =====================

class TestSpectralConv2d:
    def test_output_shape(self):
        layer = SpectralConv2d(12, 12, modes1=4, modes2=4)
        x = torch.randn(2, 12, 101, 40)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 12, 101, 40)

    def test_output_shape_batch_1(self):
        layer = SpectralConv2d(12, 12, modes1=4, modes2=4)
        x = torch.randn(1, 12, 101, 40)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (1, 12, 101, 40)

    def test_output_shape_small_grid(self):
        """Grid smaller than modes — min(modes, N_fft) guard is exercised."""
        layer = SpectralConv2d(4, 4, modes1=16, modes2=16)
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 4, 8, 8)

    def test_output_finite(self):
        layer = SpectralConv2d(8, 8, modes1=4, modes2=4)
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            y = layer(x)
        assert torch.all(torch.isfinite(y))

    def test_different_in_out_channels(self):
        layer = SpectralConv2d(4, 8, modes1=4, modes2=4)
        x = torch.randn(2, 4, 16, 16)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 8, 16, 16)

    def test_deterministic(self):
        layer = SpectralConv2d(4, 4, modes1=4, modes2=4)
        layer.eval()
        x = torch.randn(1, 4, 16, 16)
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x)
        assert torch.equal(y1, y2)


# ===================== FNO2d =====================

class TestFNO2d:
    def test_output_shape_standard(self, small_fno):
        x = torch.randn(2, 101, 40, 15)
        with torch.no_grad():
            y = small_fno(x)
        assert y.shape == (2, 101, 40, 1)

    def test_output_shape_batch_1(self, small_fno):
        x = torch.randn(1, 101, 40, 15)
        with torch.no_grad():
            y = small_fno(x)
        assert y.shape == (1, 101, 40, 1)

    def test_output_shape_batch_16(self, small_fno):
        x = torch.randn(16, 101, 40, 15)
        with torch.no_grad():
            y = small_fno(x)
        assert y.shape == (16, 101, 40, 1)

    def test_output_finite(self, small_fno):
        x = torch.randn(2, 101, 40, 15)
        with torch.no_grad():
            y = small_fno(x)
        assert torch.all(torch.isfinite(y))

    def test_padding_removed(self, small_fno):
        """Nx in output matches input despite internal padding."""
        x = torch.randn(1, 101, 40, 15)
        with torch.no_grad():
            y = small_fno(x)
        assert y.shape[1] == 101  # Nx preserved

    def test_parameter_count_positive(self, small_fno):
        n_params = sum(p.numel() for p in small_fno.parameters())
        assert n_params > 0

    def test_gradients_flow(self, small_fno):
        x = torch.randn(2, 101, 40, 15)
        y = small_fno(x)
        loss = y.mean()
        loss.backward()
        for name, p in small_fno.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_width(self):
        model = FNO2d(modes1=2, modes2=2, width=32, in_channels=15)
        x = torch.randn(1, 101, 40, 15)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 101, 40, 1)

    def test_eval_mode_deterministic(self, small_fno):
        small_fno.eval()
        x = torch.randn(1, 101, 40, 15)
        with torch.no_grad():
            y1 = small_fno(x)
            y2 = small_fno(x)
        assert torch.equal(y1, y2)
