import copy
import csv
import json
from itertools import product
from pathlib import Path
import signal

import numpy as np
import pytest
import torch
import yaml

from data.generate_dataset import build_base_setup, generate_sim_data
from problems.registry import get_problem
import src.operators.train_pino as pino
from src.operators.train import load_config
from src.physics.fv_residual import build_homogeneous_cn_geom, cn_step_residual
from tests.test_cvit_pino import config, save_inputs
from tests.test_pcvit_mg_online import assert_same_state, load_checkpoint


def prescribed_inputs(n=4):
    setup = build_base_setup(512, 1, nx=n, ny=n, t_final=.3)
    spec = get_problem('forcing_homogeneous')
    params = spec.sample_sim_params(np.random.default_rng(0), np.random.default_rng(1),
                                   setup['grids'], setup['time_cfg'])
    return dict(spec=spec, params=np.array(params, dtype=object), x=setup['grids']['x_grid'],
                y=setup['grids']['y_grid'], dt=.005, t_final=.3, ramp_seconds=.01,
                mu_global=295, sigma_global=10), setup


@pytest.fixture
def homogeneous_files(tmp_path):
    inputs, _ = prescribed_inputs()
    data = tmp_path / 'data'
    data.mkdir()
    norm = save_inputs(data, inputs)
    np.save(data / 't_grid.npy', [0, .005, .3])
    np.save(data / 'meta.npy', dict(problem_version=inputs['spec'].problem_version, ic_mode='uniform'))
    gates = []
    for stage in ('phase0_direct_state', 'phase4_direct_state_mg', 'phase4_fixed_cvit_autoregressive_prefix_mg'):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / 'final_metrics.json').write_text(json.dumps(dict(stage=stage, passed=True)))
        gates.append(directory)
    return inputs, data, norm, gates


def online_config(model='cvit'):
    cfg = config(model, residual_mode='mg')
    cfg['benchmark']['name'] = 'forcing_homogeneous'
    cfg['physics_test']['pino'].update(online_sampling='per_update', compact_metrics=True,
        online_updates=2, online_validate_every=1, full_validate_every=2,
        prefix_steps=2, validation_rollout_steps=4, batch_size=2,
        validation_cases=16, validation_batch_size=4, log_every=1,
        startup_profile_updates=[])
    return cfg


def test_sampling_matches_forcing_without_material_interface():
    inputs, setup = prescribed_inputs()
    reference = get_problem('forcing').sample_sim_params(
        np.random.default_rng(0), np.random.default_rng(1), setup['grids'], setup['time_cfg'])
    for actual, expected in zip(inputs['params'], reference):
        expected = {k: v for k, v in expected.items() if k not in {'R_c', 'interface_x'}}
        assert_same_state(actual, expected)
    inputs['spec'].validate_schema(inputs['params'], np.arange(512))
    assert {(p['temporal_family'], p['spatial_family']) for p in inputs['params']} == set(
        product(inputs['spec'].temporal_families, inputs['spec'].spatial_families))
    with pytest.raises(ValueError, match='admits only'):
        get_problem('diffusion_forcing_single').validate_schema(inputs['params'], np.arange(512))
    for mutation in ({'R_c': 0.0}, {'T0': np.full((4, 4), 301)}, {'ic_family': 'hot_spot_2d'}):
        changed = dict(inputs['params'][0], **mutation)
        with pytest.raises(ValueError):
            inputs['spec'].validate_schema(np.array([changed], dtype=object), [0])


def test_validation_balances_forcing_combinations_and_fixed_stream_visits_all():
    inputs, _ = prescribed_inputs()
    cfg = online_config()
    cfg['physics_test']['pino']['validation_cases'] = 32
    stream = pino.matched_stream(cfg, inputs, 'online')
    assert_same_state(stream, pino.matched_stream(cfg, inputs, 'online'))
    pairs = [(p['temporal_family'], p['spatial_family']) for p in stream['validation']]
    assert len(set(pairs)) == 16 and all(pairs.count(pair) == 2 for pair in set(pairs))
    a, b = pino.online_generators(42), pino.online_generators(42)
    first = pino.draw_online_batch(inputs, a, 16)
    assert_same_state(first, pino.draw_online_batch(inputs, b, 16))
    second = pino.draw_online_batch(inputs, a, 16)
    assert [p['temporal_params'] for p in first] != [p['temporal_params'] for p in second]
    b['ic_params'].random(1000)
    assert_same_state(second, pino.draw_online_batch(inputs, b, 16))
    cfg['physics_test']['pino'].update(online_sampling='precomputed', fixed_updates=16)
    fixed = pino.matched_stream(cfg, inputs, 'fixed')
    assert len(fixed['problems']) == 16
    assert set(fixed['ids'].ravel()) == set(range(16))
    assert set(fixed['validation_ids']) <= set(fixed['train_ids'])
    cfg['physics_test']['pino'].update(online_sampling='per_update', validation_cases=4)
    with pytest.raises(ValueError, match='16 forcing combinations'):
        pino.matched_stream(cfg, inputs, 'online')
    with pytest.raises(ValueError, match='missing'):
        inputs['spec'].fixed_case_ids(inputs['params'], [0])


def test_solver_and_physics_residual_agree_for_every_forcing_combination():
    inputs, setup = prescribed_inputs(20)
    params = pino.draw_online_batch(inputs, pino.online_generators(1042), 16,
        inputs['spec'].validation_assignments(16, np.random.default_rng(2)))
    geom = build_homogeneous_cn_geom(inputs['x'], inputs['y'], 1, .005, sigma_global=10)
    for p in params:
        solver = inputs['spec'].configure_solver(p, setup['base_kwargs'])
        assert len(solver.layers) == 1 and solver.layers[0].k == 1
        assert not solver.interface_R
        previous = p['T0'].astype(np.float64)
        for n in range(60):
            following = solver.cn_step(previous, n * .005)
            bc = pino.make_boundary([p], [n], inputs, torch.device('cpu'), torch.float64)
            residual = cn_step_residual(torch.tensor((previous - 295) / 10)[None],
                torch.tensor((following - 295) / 10)[None], geom, bc)
            assert residual.abs().max() < 1e-10, (p['temporal_family'], p['spatial_family'], n)
            previous = following


def test_generated_dataset_normalization_and_versioned_reload(tmp_path):
    generate_sim_data(num_sims=32, save_dir=tmp_path, benchmark='forcing_homogeneous', nx=10, ny=10)
    normalization = tmp_path / 'normalization.json'
    pino.freeze_development_normalization(tmp_path, normalization)
    loaded = pino.load_pino_inputs(tmp_path, normalization, 'forcing_homogeneous')
    assert loaded['spec'].name == 'forcing_homogeneous'
    assert loaded['meta']['ic_families'] == ['uniform_2d']
    assert loaded['meta']['ic_mode'] == 'uniform'
    assert loaded['sigma_global'] > 0
    labels = np.load(tmp_path / 'trajectories.npy')
    assert labels.shape == (32, 31, 10, 10)
    np.testing.assert_array_equal(labels[:, 0], 300.)
    np.testing.assert_array_equal(labels[:, :, -1], 300.)


def test_fno_step_contract_preserves_nonuniform_signed_and_zero_flux():
    spec = get_problem('forcing_homogeneous')
    state = torch.randn(3, 4, 4, requires_grad=True)
    x, y = torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing='ij')
    coords = torch.stack((x, y), -1).reshape(1, -1, 2)
    profiles = torch.tensor([[0., 2., -1., 0.], [0., 0., 0., 0.], [-2., -1., 0., 3.]])
    fields = spec.build_step_inputs(state, coords, profiles, .005, .3)
    torch.testing.assert_close(fields['spatial'][..., 3], profiles[:, None].expand(-1, 4, -1))
    torch.testing.assert_close(fields['forcing_seq'][..., 1:], torch.ones(3, 128, 2))
    fields['spatial'][..., 0].sum().backward()
    torch.testing.assert_close(state.grad, torch.ones_like(state))


@pytest.mark.parametrize('model', ['cvit', 'fno'])
def test_online_training_validation_and_resume(homogeneous_files, tmp_path, monkeypatch, model):
    inputs, data, norm, gates = homogeneous_files
    cfg = online_config(model)
    if model == 'cvit':
        cfg['training']['optimizer'] = 'SOAP'
    (gates[2] / 'final_metrics.json').write_text(json.dumps(dict(
        stage=f'phase4_fixed_{model}_autoregressive_prefix_mg', passed=True)))
    baseline, restarted = tmp_path / 'baseline', tmp_path / 'restart'
    def run(output):
        return pino.run_screen(cfg, data, output, gates[1], 'online', gates[2], norm, smoke=True)
    assert run(baseline)['status'] == 'complete'
    original = pino.autoregressive_backward
    calls = []
    def interrupt(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 1:
            signal.raise_signal(signal.SIGUSR1)
        return result
    monkeypatch.setattr(pino, 'autoregressive_backward', interrupt)
    assert run(restarted)['status'] == 'interrupted'
    monkeypatch.setattr(pino, 'autoregressive_backward', original)
    assert pino.resume_screen(restarted)['status'] == 'complete'
    a, b = load_checkpoint(baseline), load_checkpoint(restarted)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'sampler_rng_states'):
        assert_same_state(a[key], b[key])
    summary = json.loads((baseline / 'mg/validation_000002.json').read_text())
    assert len(summary['forcing_combinations']) == 16
    assert len(summary['temporal_families']) == len(summary['spatial_families']) == 4
    assert set(summary['ic_families']) == {'uniform_2d'}
    assert len(summary['lead_times']) == 4
    assert np.isfinite(summary['full_trajectory_rmse_K'])
    with (baseline / 'mg/val_pairs.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert {r['benchmark'] for r in rows} == {'forcing_homogeneous'}
    assert {r['temporal_family'] for r in rows} == set(inputs['spec'].temporal_families)
    assert any(r['frequency'] == '' for r in rows)
    frozen = yaml.safe_load((baseline / 'config_used.yaml').read_text())
    assert 'problems/forcing_homogeneous.py' in frozen['physics_test']['protocol']['source_sha256']


def test_fixed_exact_screen_and_input_only_loader(homogeneous_files, tmp_path, monkeypatch):
    inputs, data, norm, gates = homogeneous_files
    load = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path).name != 'trajectories.npy'
        return load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    cfg = config(residual_mode='autoregressive')
    cfg['benchmark']['name'] = 'forcing_homogeneous'
    result = pino.run_screen(cfg, data, tmp_path / 'fixed', gates[0], normalization_path=norm, smoke=True)
    assert np.isfinite(result['controls']['exact']['trajectory_rmse_K'])
    assert not result['passed']
    with pytest.raises(ValueError, match='problem_version'):
        pino.load_pino_inputs(data, norm)
    with pytest.raises(ValueError, match='does not support'):
        pino.load_pino_inputs(data, norm, 'forcing')


def test_online_gate_rejects_other_benchmark(homogeneous_files, tmp_path):
    _, data, norm, gates = homogeneous_files
    cfg = online_config()
    old = copy.deepcopy(cfg)
    old['benchmark']['name'] = 'diffusion_forcing_single'
    (gates[2] / 'config_used.yaml').write_text(yaml.safe_dump(old))
    with pytest.raises(ValueError, match='fixed-case benchmark'):
        pino.run_screen(cfg, data, tmp_path / 'wrong_gate', gates[1], 'online', gates[2], norm)


def test_benchmark_config_uses_local_full_horizon_defaults(monkeypatch):
    monkeypatch.setenv('BENCHMARK', 'forcing_homogeneous')
    cfg = load_config()
    assert cfg['benchmark']['name'] == 'forcing_homogeneous'
    assert cfg['physics_test']['pino']['prefix_steps'] == 60
    assert cfg['physics_test']['pino']['validation_cases'] == 32
    assert cfg['physics_test']['pino']['batch_size'] == 1
    assert cfg['training']['optimizer'] == 'SOAP'
    assert not cfg['physics_test']['pino']['gate_compare_first_step_baselines']


def test_quiet_first_step_gate_keeps_absolute_accuracy_requirement():
    settings = config()['physics_test']['pino']
    metrics = dict(trajectory_rmse_K=.1, trajectory_sigma_error_percent=1.,
                   first_step_rmse_K=.01, first_step_sigma_error_percent=.1,
                   copy_trajectory_rmse_K=2., equilibrium_trajectory_rmse_K=2.,
                   copy_first_step_rmse_K=0., equilibrium_first_step_rmse_K=0.)
    assert not pino.cvit_gate(None, metrics, settings, 'fixed')['passed']
    settings['gate_compare_first_step_baselines'] = False
    gate = pino.cvit_gate(None, metrics, settings, 'fixed')
    assert gate['passed'] and gate['baseline_comparison_kinds'] == ['trajectory']
    metrics['first_step_sigma_error_percent'] = settings['fixed_max_first_step_sigma_error_percent'] + 1
    assert not pino.cvit_gate(None, metrics, settings, 'fixed')['passed']
