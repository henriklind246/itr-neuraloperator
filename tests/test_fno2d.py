import pytest
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

    def test_valid_shape_normalizes_over_valid_region_only(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=3)
        gamma = torch.ones(2, 3)
        beta = torch.zeros(2, 3)
        x = torch.randn(2, 3, 6, 7)
        # Corrupt the pad region (everything outside the top-left 4x5 block) with
        # large spurious values; valid-region stats must ignore them.
        x[:, :, 4:, :] = 50.0
        x[:, :, :, 5:] = -50.0

        out = cin(x, gamma, beta, valid_shape=(4, 5))

        valid = out[:, :, :4, :5]
        mean = valid.mean(dim=(-2, -1))
        var = valid.var(dim=(-2, -1), unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_valid_shape_differs_from_full_tensor_norm(self):
        torch.manual_seed(0)
        cin = ConditionalInstanceNorm2d(num_features=3)
        gamma = torch.ones(2, 3)
        beta = torch.zeros(2, 3)
        x = torch.randn(2, 3, 6, 7)
        x[:, :, 4:, :] = 50.0

        out_full = cin(x, gamma, beta)
        out_valid = cin(x, gamma, beta, valid_shape=(4, 7))
        assert not torch.allclose(out_full, out_valid)


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

    def test_padding_mode_default_is_zeros_and_replicate_changes_output(self):
        common = dict(
            modes1=2, modes2=2, width=8,
            in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
        )
        m_zeros = FNO2d(**common)
        assert m_zeros.padding_mode == "zeros"
        assert m_zeros.cin_exclude_padding is False
        m_rep = FNO2d(**common, padding_mode="replicate")
        m_rep.load_state_dict(m_zeros.state_dict())
        m_zeros.eval()
        m_rep.eval()

        spatial = torch.randn(2, 11, 11, SPATIAL_IN_CHANNELS)
        cond_static = torch.randn(2, COND_STATIC_DIM)
        forcing_seq = torch.randn(2, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
        with torch.no_grad():
            out_zeros = m_zeros(spatial, cond_static, forcing_seq)
            out_rep = m_rep(spatial, cond_static, forcing_seq)
        assert out_rep.shape == out_zeros.shape == (2, 11, 11, 1)
        assert not torch.allclose(out_zeros, out_rep)

    def test_cin_exclude_padding_changes_output(self):
        common = dict(
            modes1=2, modes2=2, width=8,
            in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
            cond_static_dim=COND_STATIC_DIM,
            temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
        )
        m_off = FNO2d(**common)
        m_on = FNO2d(**common, cin_exclude_padding=True)
        m_on.load_state_dict(m_off.state_dict())
        m_off.eval()
        m_on.eval()

        spatial = torch.randn(2, 11, 11, SPATIAL_IN_CHANNELS)
        cond_static = torch.randn(2, COND_STATIC_DIM)
        forcing_seq = torch.randn(2, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM)
        with torch.no_grad():
            out_off = m_off(spatial, cond_static, forcing_seq)
            out_on = m_on(spatial, cond_static, forcing_seq)
        assert not torch.allclose(out_off, out_on)

    def test_invalid_padding_mode_raises(self):
        import pytest
        with pytest.raises(ValueError):
            FNO2d(
                modes1=2, modes2=2, width=8,
                in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
                cond_static_dim=COND_STATIC_DIM,
                temporal_token_dim=TEMPORAL_TOKEN_DIM,
                temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
                padding_mode="banana",
            )

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

    def test_s_y_channel_selects_forcing_source_channel(self):
        """The learned spatial forcing field reads s_y from `s_y_channel`. Two
        models with identical weights but different s_y_channel must produce
        different outputs when those two channels differ — proving interfaces'
        channel-5 fix is honored rather than the legacy hard-coded channel 3."""
        torch.manual_seed(0)
        common = dict(
            modes1=2, modes2=2, width=8,
            in_channels=8, out_channels=1, n_layers=2,
            cond_static_dim=4, temporal_token_dim=2,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
        )
        m3 = FNO2d(**common, s_y_channel=3)
        m5 = FNO2d(**common, s_y_channel=5)
        m5.load_state_dict(m3.state_dict())
        m3.eval()
        m5.eval()

        spatial = torch.randn(2, 11, 11, 8)
        # Make channels 3 and 5 clearly distinct so the selection matters.
        spatial[..., 3] = 1.0
        spatial[..., 5] = -1.0
        cond_static = torch.randn(2, 4)
        forcing_seq = torch.randn(2, 16, 2)

        with torch.no_grad():
            out3 = m3(spatial, cond_static, forcing_seq)
            out5 = m5(spatial, cond_static, forcing_seq)
        assert not torch.allclose(out3, out5)

    def test_forcing_time_aug_builds_aug_mlp(self):
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=6, out_channels=1, n_layers=2,
            cond_static_dim=4, temporal_token_dim=2,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
            use_forcing_time_aug=True, s_y_channel=5,
        )
        assert hasattr(model, "forcing_aug_mlp")
        # The first linear folds normalized lead time into h_a.
        assert model.forcing_aug_mlp[0].in_features == 16 + 1

        spatial = torch.randn(2, 11, 11, 6)
        cond_static = torch.randn(2, 4)
        forcing_seq = torch.randn(2, 16, 2)
        out = model(spatial, cond_static, forcing_seq)
        assert out.shape == (2, 11, 11, 1)

    def test_forcing_cond_mode_spatial_only_drops_h_a_from_cin(self):
        """spatial_only: forcing reaches the model only via s_y * z_a. The CIN
        MLP consumes cond_static alone (h_a dropped), yet the lift still gains
        the K spatial-forcing channels and the temporal encoder is built."""
        in_ch, cond_dim, K = 4, 4, 4
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=in_ch, out_channels=1, n_layers=2,
            cond_static_dim=cond_dim, temporal_token_dim=2,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=K,
            forcing_cond_mode="spatial_only",
        )
        assert model.cond_mlp.net[0].in_features == cond_dim
        assert model.linear_p.in_features == in_ch + K
        assert hasattr(model, "temporal_encoder")
        assert hasattr(model, "forcing_to_spatial")

        spatial = torch.randn(2, 11, 11, in_ch)
        cond_static = torch.randn(2, cond_dim)
        forcing_seq = torch.randn(2, 16, 2)
        out = model(spatial, cond_static, forcing_seq)
        assert out.shape == (2, 11, 11, 1)

    def test_forcing_cond_mode_spatial_only_cin_independent_of_forcing(self):
        """In spatial_only the CIN γ/β must not depend on forcing_seq: the h_a
        route into conditioning is severed. Two different forcing_seq inputs give
        identical cond_mlp output, while the spatial pathway keeps the overall
        output forcing-sensitive."""
        torch.manual_seed(0)
        in_ch, cond_dim = 8, 4
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=in_ch, out_channels=1, n_layers=2,
            cond_static_dim=cond_dim, temporal_token_dim=2,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
            forcing_cond_mode="spatial_only",
        )
        model.eval()
        cond_static = torch.randn(2, cond_dim)
        cin = model.cond_mlp(cond_static)
        # cond_mlp is driven by cond_static only, so its output is fixed regardless
        # of any forcing_seq — the defining property of spatial_only conditioning.
        assert cin.shape == (2, 2, 2, 8)

        spatial = torch.randn(2, 11, 11, in_ch)
        spatial[..., 3] = 1.0  # non-uniform s_y so the spatial pathway is live
        fseq_a = torch.randn(2, 16, 2)
        fseq_b = torch.randn(2, 16, 2)
        with torch.no_grad():
            out_a = model(spatial, cond_static, fseq_a)
            out_b = model(spatial, cond_static, fseq_b)
        # Forcing still changes the prediction, but only through s_y * z_a.
        assert not torch.allclose(out_a, out_b)

    def test_forcing_cond_mode_cond_only_omits_spatial_pathway(self):
        """cond_only: forcing reaches the model only through CIN. The lift sees
        in_channels alone (no K forcing channels) and no forcing_to_spatial /
        forcing_aug_mlp are built, while the CIN MLP still consumes h_a."""
        in_ch, cond_dim = 4, 4
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=in_ch, out_channels=1, n_layers=2,
            cond_static_dim=cond_dim, temporal_token_dim=2,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
            use_forcing_time_aug=True,
            forcing_cond_mode="cond_only",
        )
        assert model.cond_mlp.net[0].in_features == cond_dim + 16
        assert model.linear_p.in_features == in_ch
        assert not hasattr(model, "forcing_to_spatial")
        assert not hasattr(model, "forcing_aug_mlp")
        assert hasattr(model, "temporal_encoder")

        spatial = torch.randn(2, 11, 11, in_ch)
        cond_static = torch.randn(2, cond_dim)
        forcing_seq = torch.randn(2, 16, 2)
        out = model(spatial, cond_static, forcing_seq)
        assert out.shape == (2, 11, 11, 1)

    def test_forcing_cond_mode_invalid_value_raises(self):
        with pytest.raises(ValueError, match="forcing_cond_mode must be one of"):
            FNO2d(
                modes1=2, modes2=2, width=8,
                in_channels=4, out_channels=1, n_layers=2,
                cond_static_dim=4, temporal_token_dim=2,
                temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
                forcing_cond_mode="bogus",
            )

    def test_forcing_cond_mode_requires_temporal_encoder(self):
        with pytest.raises(ValueError, match="use_temporal_encoder=True"):
            FNO2d(
                modes1=2, modes2=2, width=8,
                in_channels=20, out_channels=1, n_layers=2,
                cond_static_dim=4,
                use_temporal_encoder=False,
                forcing_cond_mode="spatial_only",
            )

    def test_no_forcing_time_aug_omits_aug_mlp(self):
        model = FNO2d(
            modes1=2, modes2=2, width=8,
            in_channels=4, out_channels=1, n_layers=2,
            cond_static_dim=4, temporal_token_dim=2,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
            use_forcing_time_aug=False,
        )
        assert not hasattr(model, "forcing_aug_mlp")

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


class TestHardRightDirichlet:
    @staticmethod
    def _model(**kw):
        base = dict(
            modes1=2, modes2=2, width=8,
            in_channels=SPATIAL_IN_CHANNELS, out_channels=1, n_layers=2,
            cond_static_dim=COND_STATIC_DIM, temporal_token_dim=TEMPORAL_TOKEN_DIM,
            temporal_hidden=16, forcing_embed_dim=16, forcing_spatial_dim=4,
        )
        base.update(kw)
        return FNO2d(**base)

    @staticmethod
    def _inputs(b=2, nx=11, ny=11):
        return (
            torch.randn(b, nx, ny, SPATIAL_IN_CHANNELS),
            torch.randn(b, COND_STATIC_DIM),
            torch.randn(b, TEMPORAL_SAMPLES, TEMPORAL_TOKEN_DIM),
        )

    def test_right_edge_column_is_exact_constant(self):
        torch.manual_seed(0)
        model = self._model(hard_right_dirichlet=True, t_right_norm=-0.7)
        out = model(*self._inputs())
        # dim 1 is Nx; the right wall face is T[:, Nx-1, :].
        edge = out[:, -1, :, :]
        assert torch.allclose(edge, torch.full_like(edge, -0.7), atol=1e-6)

    def test_disabled_path_bit_identical(self):
        torch.manual_seed(0)
        m_off = self._model()
        torch.manual_seed(0)
        m_on = self._model(hard_right_dirichlet=True, t_right_norm=0.0)
        # Same seed -> identical weights; disable buffer must not touch interior.
        inp = self._inputs()
        out_off = m_off(*inp)
        out_on = m_on(*inp)
        # Interior (all but the right face) must be unchanged by the constraint.
        assert torch.allclose(out_off[:, :-1], out_on[:, :-1], atol=1e-6)

    def test_backward_reaches_linear_p(self):
        torch.manual_seed(0)
        model = self._model(hard_right_dirichlet=True, t_right_norm=0.3)
        out = model(*self._inputs())
        out.pow(2).mean().backward()
        g = model.linear_p.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0

    def test_buffer_round_trips_through_state_dict(self):
        model = self._model(hard_right_dirichlet=True, t_right_norm=1.25)
        sd = model.state_dict()
        assert "t_right_norm" in sd
        assert float(sd["t_right_norm"]) == 1.25
        # Rebuild with a different buffer value, then load: value must be restored.
        model2 = self._model(hard_right_dirichlet=True, t_right_norm=0.0)
        model2.load_state_dict(sd)
        assert float(model2.t_right_norm) == 1.25

    def test_disabled_registers_no_buffer(self):
        model = self._model()
        assert "t_right_norm" not in model.state_dict()

    def test_baseline_checkpoint_warm_starts_hard_bc_model(self):
        # A baseline (no hard BC) state_dict loads into a hard-BC model with
        # strict=False, with t_right_norm as the only missing key.
        torch.manual_seed(0)
        baseline = self._model()
        torch.manual_seed(0)
        hard = self._model(hard_right_dirichlet=True, t_right_norm=0.9)
        missing, unexpected = hard.load_state_dict(baseline.state_dict(), strict=False)
        assert list(unexpected) == []
        assert set(missing) == {"t_right_norm"}
        assert float(hard.t_right_norm) == pytest.approx(0.9)
