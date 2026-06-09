import torch

from src.operators.fno2d import ConditionalInstanceNorm2d, FNO2d, TemporalForcingEncoder

SPATIAL_IN_CHANNELS = 20
COND_STATIC_DIM = 23
TEMPORAL_TOKEN_DIM = 5
TEMPORAL_SAMPLES = 64


class TestConditionalInstanceNorm2d:
    def test_matches_instance_norm_and_conditioning_affine(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=8)
        x = torch.randn(2, 8, 12, 12, requires_grad=True)
        gamma = torch.randn(2, 8, requires_grad=True)
        beta = torch.randn(2, 8, requires_grad=True)

        y_actual = cin(x, gamma, beta)
        y_norm = cin.norm(x)
        y_expected = gamma[:, :, None, None] * y_norm + beta[:, :, None, None]

        assert torch.allclose(y_actual, y_expected, atol=1e-6)

        grads_actual = torch.autograd.grad(y_actual.sum(), (x, gamma, beta), retain_graph=True)
        grads_expected = torch.autograd.grad(y_expected.sum(), (x, gamma, beta), retain_graph=True)
        for grad_actual, grad_expected in zip(grads_actual, grads_expected):
            assert torch.allclose(grad_actual, grad_expected, atol=1e-5)

    def test_full_tensor_has_instance_norm_statistics(self):
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

    def test_valid_shape_is_ignored_for_native_instance_norm(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=3)
        gamma = torch.ones(2, 3)
        beta = torch.zeros(2, 3)
        x = torch.randn(2, 3, 4, 5)

        out_full = cin(x, gamma, beta)
        out_with_shape = cin(x, gamma, beta, valid_shape=(3, 4))

        assert torch.equal(out_full, out_with_shape)


class TestTemporalForcingEncoder:
    def test_output_shape(self):
        torch.manual_seed(0)
        enc = TemporalForcingEncoder(token_dim=TEMPORAL_TOKEN_DIM, hidden=32, embed_dim=16)
        z = torch.randn(3, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
        h = enc(z)
        assert h.shape == (3, 16)


class TestFNO2d:
    def test_forward_returns_channels_last_output(self):
        model = FNO2d(
            modes1=2,
            modes2=2,
            width=8,
            in_channels=SPATIAL_IN_CHANNELS,
            out_channels=1,
            n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16,
            forcing_embed_dim=16,
            forcing_spatial_dim=4,
        )
        spatial = torch.randn(2, 11, 11, SPATIAL_IN_CHANNELS)
        cond_static = torch.randn(2, COND_STATIC_DIM)
        forcing_seq = torch.randn(2, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)

        out = model(spatial, cond_static, forcing_seq)

        assert out.shape == (2, 11, 11, 1)

    def test_padding_reference_resolution_scales_pad_cells(self):
        model = FNO2d(
            modes1=2,
            modes2=2,
            width=8,
            in_channels=SPATIAL_IN_CHANNELS,
            out_channels=1,
            n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16,
            forcing_embed_dim=16,
            forcing_spatial_dim=4,
            padding_reference_resolution=100,
        )

        assert model._padding_for_shape(100, 100) == (8, 8)
        assert model._padding_for_shape(200, 200) == (16, 16)
        assert model._padding_for_shape(256, 256) == (21, 21)

    def test_default_padding_keeps_fixed_cell_count(self):
        model = FNO2d(
            modes1=2,
            modes2=2,
            width=8,
            in_channels=SPATIAL_IN_CHANNELS,
            out_channels=1,
            n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16,
            forcing_embed_dim=16,
            forcing_spatial_dim=4,
        )

        assert model._padding_for_shape(100, 100) == (8, 8)
        assert model._padding_for_shape(256, 256) == (8, 8)

    def test_cond_mlp_input_width_encoder_on(self):
        """With the temporal encoder on, the CIN MLP consumes
        cond_static_dim + forcing_embed_dim."""
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
        )
        first_linear = model.cond_mlp.net[0]
        assert first_linear.in_features == COND_STATIC_DIM + 16
        # lift sees in_channels + forcing_spatial_dim
        assert model.linear_p.in_features == SPATIAL_IN_CHANNELS + 4
        assert hasattr(model, "temporal_encoder")
        assert hasattr(model, "forcing_to_spatial")

    def test_forward_encoder_off_no_forcing_seq(self):
        """Encoder-off path (source benchmark): forward works with
        forcing_seq=None, lift sees in_channels alone, CIN consumes
        cond_static alone."""
        in_ch = 20
        cond_dim = 8
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=in_ch, out_channels=1, n_layers=2,
            cond_static_dim=cond_dim,
            use_temporal_encoder=False,
        )
        assert not hasattr(model, "temporal_encoder")
        assert not hasattr(model, "forcing_to_spatial")
        assert model.linear_p.in_features == in_ch
        assert model.cond_mlp.net[0].in_features == cond_dim

        spatial = torch.randn(2, 11, 11, in_ch)
        cond_static = torch.randn(2, cond_dim)
        out = model(spatial, cond_static)
        assert out.shape == (2, 11, 11, 1)

    def test_near_identity_init_at_step_zero(self):
        """Soft identity init: head weights small-random (std=1e-3), bias at γ=1, β=0.

        Per-element deviation can occasionally exceed 0.05; the mean-deviation is the
        stable property that confirms the model starts near an unconditioned FNO while
        still allowing gradient flow through the conditioning path.
        """
        torch.manual_seed(0)
        model = FNO2d(
            modes1=2,
            modes2=2,
            width=8,
            in_channels=SPATIAL_IN_CHANNELS,
            out_channels=1,
            n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16,
            forcing_embed_dim=16,
            forcing_spatial_dim=4,
        )
        cond_static = torch.randn(4, COND_STATIC_DIM)
        forcing_seq = torch.randn(4, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
        h_a = model.temporal_encoder(forcing_seq)
        cond_full = torch.cat([cond_static, h_a], dim=-1)
        cin_params = model.cond_mlp(cond_full)
        gamma = cin_params[:, :, 0, :]
        beta = cin_params[:, :, 1, :]
        assert (gamma - 1.0).abs().mean() < 0.01
        assert beta.abs().mean() < 0.01
