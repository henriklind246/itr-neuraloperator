import copy
import csv
import json
from itertools import product
import os
from pathlib import Path
import signal
import subprocess

import numpy as np
import pytest
import torch
import yaml

from data.generate_dataset import build_base_setup, generate_sim_data
from problems.registry import get_problem
from scripts.run_train_fixed import _apply_override, _parse_override_value
import src.operators.train_pino as pino
from src.operators.fno2d import FNO2d
from src.operators.soap import SOAP
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


def space_time_config(control='mg', optimizer='SOAP', scheduler=None, ic_weight=0.0):
    cfg = config('fno', residual_mode='space_time')
    cfg['benchmark']['name'] = 'forcing_homogeneous'
    cfg['training']['optimizer'] = optimizer
    if scheduler is not None:
        cfg['training']['scheduler'] = scheduler
    cfg['physics_test']['pino'].update(controls=[control], online_sampling='per_update', compact_metrics=True,
        online_updates=4, online_validate_every=2, full_validate_every=4, prefix_steps=3,
        validation_rollout_steps=4, batch_size=2, validation_cases=16, validation_batch_size=8,
        log_every=1, startup_profile_updates=[], time_chunk_size=2, ic_weight=ic_weight)
    return cfg


RIGNO = dict(type='RIGNOThreePhase', warmup_fraction=.25, cosine_fraction=.5, exp_fraction=.25,
             init_lr_ratio=.1, cosine_floor_lr_ratio=.1, final_lr_ratio=.05)


@pytest.mark.parametrize('control,optimizer,scheduler,ic_weight', [
    ('mg', 'SOAP', None, 0.0), ('spatial_mg_detached', 'Adam', RIGNO, 0.0), ('raw', 'AdamW', None, 0.5)])
def test_online_space_time_fno_trains_validates_and_resumes(homogeneous_files, tmp_path, monkeypatch,
                                                            control, optimizer, scheduler, ic_weight):
    inputs, data, norm, gates = homogeneous_files
    cfg = space_time_config(control, optimizer, scheduler, ic_weight)
    gate = gates[1] if control in pino.MG_OBJECTIVES else gates[0]
    baseline, restarted = tmp_path / 'baseline', tmp_path / 'restart'
    def run(output):
        # No fixed-case screen: online space_time runs are ungated development evidence.
        return pino.run_screen(cfg, data, output, gate, 'online', None, norm, smoke=True)
    original, calls = pino.space_time_backward, []
    def record(*args):
        calls.append(args[-1])
        return original(*args)
    monkeypatch.setattr(pino, 'space_time_backward', record)
    result = run(baseline)
    assert result['status'] == 'complete' and not result['accuracy_gate_applied']
    assert set(result['controls']) == {control} and calls == [control] * 4
    def interrupt(*args):
        value = original(*args)
        signal.raise_signal(signal.SIGUSR1)
        return value
    monkeypatch.setattr(pino, 'space_time_backward', interrupt)
    assert run(restarted)['status'] == 'interrupted'
    monkeypatch.setattr(pino, 'space_time_backward', original)
    assert pino.resume_screen(restarted)['status'] == 'complete'
    a, b = (torch.load(output / control / 'checkpoint_final.pt', weights_only=False) for output in (baseline, restarted))
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'sampler_rng_states'):
        assert_same_state(a[key], b[key])
    assert (baseline / control / 'training_manifest.jsonl').read_bytes() == (
        restarted / control / 'training_manifest.jsonl').read_bytes()
    frozen = yaml.safe_load((baseline / 'config_used.yaml').read_text())
    protocol = frozen['physics_test']['protocol']
    assert frozen['training']['epochs'] == 4
    assert protocol['preconditioner'] == {control: pino.PREFIX_PRECONDITIONERS[control]}
    assert protocol['gate_policy'] == 'registered_final_update_evidence_no_accuracy_gate'
    assert 'recomputed per time chunk' in protocol['temporal_gradient_policy']
    assert (protocol['mg_settings'] is not None) == (control in pino.MG_OBJECTIVES)
    assert 'slurm/train_fno_space_time_msi.sbatch' in protocol['source_sha256']
    assert [p['path'] for p in protocol['prerequisites']] == [str((gate / 'final_metrics.json').resolve())]
    if scheduler is not None:
        assert a['scheduler_state_dict'] == {'next_epoch_index': 4}
    with (baseline / control / 'train_metrics.csv').open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames[-1] == 'ic_loss'
        rows = list(reader)
    assert [int(r['update']) for r in rows] == [1, 2, 3, 4]
    assert all(float(r['ic_loss']) > 0 for r in rows) == bool(ic_weight)
    assert np.load(baseline / 'validation_references.npy').shape == (16, 5, 4, 4)
    summary = json.loads((baseline / control / 'validation_000004.json').read_text())
    assert np.isfinite(summary['full_trajectory_rmse_K']) and len(summary['lead_times']) == 4
    assert len(summary['forcing_combinations']) == 16


@pytest.mark.parametrize('mutate,gate_index,match', [
    (lambda c: None, 0, 'phase4_direct_state_mg'),
    (lambda c: c['physics_test'].update(model_type='cvit'), 1, 'one control of an online space_time FNO run'),
    (lambda c: c['physics_test']['pino'].update(controls=['raw', 'mg']), 1, 'one control of an online space_time FNO run'),
    (lambda c: c['model']['parameters'].update(hard_right_dirichlet=False), 1, 'hard_right_dirichlet'),
    (lambda c: c['model']['parameters'].update(dropout=.1), 1, 'dropout'),
    (lambda c: c['physics_test']['pino'].update(time_chunk_size=0), 1, 'time_chunk_size'),
    (lambda c: c['physics_test']['pino'].update(validation_seed=42), 1, 'independent seed'),
])
def test_online_space_time_rejects_invalid_runs_before_creating_artifacts(homogeneous_files, tmp_path,
                                                                         mutate, gate_index, match):
    _, data, norm, gates = homogeneous_files
    cfg = space_time_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        pino.run_screen(cfg, data, tmp_path / 'run', gates[gate_index], 'online', None, norm, smoke=True)
    assert not (tmp_path / 'run').exists()


def test_online_space_time_scientific_runs_need_the_benchmark_grid_not_a_minimum_budget(homogeneous_files, tmp_path):
    _, data, norm, gates = homogeneous_files
    with pytest.raises(ValueError, match='100x100 grid'):
        pino.run_screen(space_time_config(), data, tmp_path / 'run', gates[1], 'online', None, norm, smoke=False)


def test_space_time_launcher_defaults_overrides_and_resume(tmp_path, monkeypatch):
    script = Path('slurm/train_fno_space_time_msi.sbatch').resolve()
    subprocess.run(['bash', '-n', script], check=True)
    text = script.read_text()
    for directive in ['--partition=msigpu', '--gres=gpu:a100:1', '--time=08:00:00', '--signal=USR1@120']:
        assert directive in text
    project = tmp_path / 'project with spaces'
    project.mkdir()
    capture = tmp_path / 'arguments'
    prologue = '''module() { :; }
conda() { :; }
nvidia-smi() { :; }
srun() { printf '%s\\0' "$@" > "$CAPTURE"; }
export -f module conda nvidia-smi srun
exec bash "$LAUNCHER" "$@"
'''
    # A clean environment keeps the caller's DATA_DIR, CONTROL, ... out of the defaults.
    base = dict(PATH=os.environ['PATH'], HOME=str(tmp_path), SLURM_JOB_ID='77', SLURM_SUBMIT_DIR=str(project),
                PERSISTENT_RUNS=str(tmp_path / 'runs'), CAPTURE=str(capture), LAUNCHER=str(script))

    def launch(*extra, **env):
        subprocess.run(['bash', '-c', prologue, 'launcher', *extra], env=dict(base, **env), check=True,
                       capture_output=True)
        return capture.read_bytes().decode().strip('\0').split('\0')

    def resolved(args):
        cfg = load_config()
        for raw in (arg for arg in args if '=' in arg):
            key, value = raw.split('=', 1)
            _apply_override(cfg, key, _parse_override_value(value))
        return cfg

    monkeypatch.setenv('BENCHMARK', 'forcing_homogeneous')
    monkeypatch.setenv('REPRESENTATION', 'temporal_encoder')
    args = launch()
    assert args[:3] == ['python', '-u', 'scripts/run_train_pino.py']
    assert args[args.index('--stage') + 1] == 'online'
    assert args[args.index('--data-dir') + 1] == str(project / 'data/forcing_homogeneous')
    assert args[args.index('--phase0-dir') + 1] == str(project / 'runs/physics_test_phase4_direct_state_mg_20261002')
    assert args[args.index('--output-dir') + 1] == str(tmp_path / 'runs/fno_st_mg_SOAP_lr0.001_bs8_n60_u5000_s42_77')
    assert '--normalization' not in args
    cfg = resolved(args)
    pino_cfg = cfg['physics_test']['pino']
    assert cfg['physics_test']['model_type'] == 'fno'
    assert (pino_cfg['residual_mode'], pino_cfg['controls'], pino_cfg['online_sampling']) == ('space_time', ['mg'], 'per_update')
    assert (pino_cfg['prefix_steps'], pino_cfg['batch_size'], pino_cfg['online_updates']) == (60, 8, 5000)
    assert (pino_cfg['time_chunk_size'], pino_cfg['validation_cases'], pino_cfg['ic_weight']) == (4, 128, 0.0)
    assert cfg['model']['parameters']['hard_right_dirichlet'] and cfg['model']['parameters']['dropout'] == 0
    model = pino.build_screen_model(cfg, prescribed_inputs(n=20)[0], torch.device('cpu'))
    assert isinstance(model, FNO2d)
    optimizer = pino.build_optimizer(cfg, model.parameters())
    assert isinstance(optimizer, SOAP) and optimizer.param_groups[0]['lr'] == pytest.approx(1e-3)
    scheduler = pino.build_scheduler(cfg, optimizer)
    assert scheduler.lr_for_update(250) == pytest.approx(1e-3 * .9)

    norm = tmp_path / 'normalization with spaces.json'
    args = launch('model.parameters.width=32', 'training.soap.precondition_frequency=5',
                  CONTROL='raw', OPTIMIZER='Adam', LEARNING_RATE='3e-4', SCHEDULER='RIGNOThreePhase',
                  BATCH_SIZE='4', NORMALIZATION_PATH=str(norm))
    assert args[args.index('--phase0-dir') + 1] == str(project / 'runs/physics_test_phase0_raw_exact_20261001')
    assert args[args.index('--normalization') + 1] == str(norm)
    assert args[-2:] == ['model.parameters.width=32', 'training.soap.precondition_frequency=5']
    cfg = resolved(args)
    assert cfg['physics_test']['pino']['controls'] == ['raw'] and cfg['physics_test']['pino']['batch_size'] == 4
    assert (cfg['training']['optimizer'], cfg['training']['learning_rate']) == ('Adam', 3e-4)
    assert cfg['training']['scheduler']['type'] == 'RIGNOThreePhase'
    assert cfg['model']['parameters']['width'] == 32 and cfg['training']['soap']['precondition_frequency'] == 5
    model = pino.build_screen_model(cfg, prescribed_inputs(n=20)[0], torch.device('cpu'))
    assert isinstance(pino.build_optimizer(cfg, model.parameters()), torch.optim.Adam)

    resume = tmp_path / 'resume with spaces'
    assert launch(RESUME_RUN_DIR=str(resume))[3:] == ['--resume', '--output-dir', str(resume)]
