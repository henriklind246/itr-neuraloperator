import copy
import csv
import json
from itertools import count, product
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
from src.operators.ss_esoap import SSESOAP
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
            signal.getsignal(getattr(signal, 'SIGUSR1', signal.SIGINT))(getattr(signal, 'SIGUSR1', signal.SIGINT), None)
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


@pytest.mark.parametrize('control,optimizer,scheduler,ic_weight,local_kernel', [
    ('mg', 'SOAP', None, 0.0, 0), ('spatial_mg_detached', 'Adam', RIGNO, 0.0, 0), ('raw', 'AdamW', None, 0.5, 0),
    ('mg', 'SSESOAP', None, 0.0, 3)])
def test_online_space_time_fno_trains_validates_and_resumes(homogeneous_files, tmp_path, monkeypatch,
                                                            control, optimizer, scheduler, ic_weight, local_kernel):
    inputs, data, norm, gates = homogeneous_files
    cfg = space_time_config(control, optimizer, scheduler, ic_weight)
    cfg['model']['parameters']['local_branch_kernel_size'] = local_kernel
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
        signal.getsignal(getattr(signal, 'SIGUSR1', signal.SIGINT))(getattr(signal, 'SIGUSR1', signal.SIGINT), None)
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
    assert frozen['model']['parameters']['local_branch_kernel_size'] == local_kernel
    local = {k: v for k, v in a['model_state_dict'].items() if k.startswith('local_layers.')}
    assert bool(local) == bool(local_kernel)
    initial = torch.load(baseline / 'initial_state.pt', weights_only=True)
    assert all(not torch.equal(value, initial[key]) for key, value in local.items())
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
    (lambda c: c['physics_test']['pino'].update(physics_warmup_updates=-1), 1, 'physics_warmup_updates'),
    (lambda c: c['physics_test']['pino'].update(physics_warmup_resample_every=0), 1, 'physics_warmup_resample_every'),
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
    assert Path(args[args.index('--data-dir') + 1]) == project / 'data/forcing_homogeneous'
    assert Path(args[args.index('--phase0-dir') + 1]) == project / 'runs/physics_test_phase4_direct_state_mg_20261002'
    assert Path(args[args.index('--output-dir') + 1]) == tmp_path / 'runs/fno_st_mg_SSESOAP_lr0.0001_bs8_n60_u10000_s42_77'
    assert '--normalization' not in args
    cfg = resolved(args)
    pino_cfg = cfg['physics_test']['pino']
    assert cfg['physics_test']['model_type'] == 'fno'
    assert (pino_cfg['residual_mode'], pino_cfg['controls'], pino_cfg['online_sampling']) == ('space_time', ['mg'], 'per_update')
    assert (pino_cfg['prefix_steps'], pino_cfg['batch_size'], pino_cfg['online_updates']) == (60, 8, 10000)
    assert (pino_cfg['time_chunk_size'], pino_cfg['validation_cases'], pino_cfg['ic_weight']) == (4, 128, 0.0)
    assert cfg['model']['parameters']['hard_right_dirichlet'] and cfg['model']['parameters']['dropout'] == 0
    model = pino.build_screen_model(cfg, prescribed_inputs(n=20)[0], torch.device('cpu'))
    assert isinstance(model, FNO2d)
    optimizer = pino.build_optimizer(cfg, model.parameters())
    assert isinstance(optimizer, SSESOAP) and optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    group = optimizer.param_groups[0]
    assert group['betas'] == (.9, .95) and group['eps'] == pytest.approx(1e-8)
    assert group['check_interval'] == 1 and group['trigger_threshold'] == .2
    assert group['warmup_steps'] == 0 and group['tau_min'] == .001
    scheduler = pino.build_scheduler(cfg, optimizer)
    assert scheduler.lr_for_update(500) == pytest.approx(1e-4 * .9)
    assert scheduler.lr_for_update(pino_cfg['online_updates']) == pytest.approx(1e-4 * .9 ** 20)

    assert (pino_cfg['physics_warmup_updates'], pino_cfg['physics_warmup_resample_every']) == (0, 1)
    assert cfg['model']['parameters']['local_branch_kernel_size'] == 0 and not hasattr(model, 'local_layers')

    norm = tmp_path / 'normalization with spaces.json'
    args = launch('model.parameters.width=32', 'training.soap.precondition_frequency=5',
                  CONTROL='raw', OPTIMIZER='Adam', LEARNING_RATE='3e-4', SCHEDULER='RIGNOThreePhase',
                  BATCH_SIZE='4', NORMALIZATION_PATH=str(norm),
                  PHYSICS_WARMUP_UPDATES='500', PHYSICS_WARMUP_RESAMPLE_EVERY='50', LOCAL_BRANCH_KERNEL='3')
    assert Path(args[args.index('--phase0-dir') + 1]) == project / 'runs/physics_test_phase0_raw_exact_20261001'
    assert args[args.index('--normalization') + 1] == str(norm)
    assert args[args.index('--output-dir') + 1].endswith('_s42_pw500x50_lc3_77')
    assert args[-2:] == ['model.parameters.width=32', 'training.soap.precondition_frequency=5']
    cfg = resolved(args)
    assert cfg['physics_test']['pino']['controls'] == ['raw'] and cfg['physics_test']['pino']['batch_size'] == 4
    assert (cfg['physics_test']['pino']['physics_warmup_updates'],
            cfg['physics_test']['pino']['physics_warmup_resample_every']) == (500, 50)
    assert (cfg['training']['optimizer'], cfg['training']['learning_rate']) == ('Adam', 3e-4)
    assert cfg['training']['scheduler']['type'] == 'RIGNOThreePhase'
    assert cfg['model']['parameters']['width'] == 32 and cfg['training']['soap']['precondition_frequency'] == 5
    assert cfg['model']['parameters']['local_branch_kernel_size'] == 3
    model = pino.build_screen_model(cfg, prescribed_inputs(n=20)[0], torch.device('cpu'))
    assert isinstance(pino.build_optimizer(cfg, model.parameters()), torch.optim.Adam)
    assert len(model.local_layers) == model.n_layers

    resume = tmp_path / 'resume with spaces'
    assert launch(RESUME_RUN_DIR=str(resume))[3:] == ['--resume', '--output-dir', str(resume)]


@pytest.mark.parametrize('warmup_updates,resample_every,expected', [
    (100, 3, [0, 3]), (2, 3, [0, 2, 3, 4, 5]), (0, 50, [0, 1, 2, 3, 4, 5])])
def test_physics_warmup_resampling_cadence_matches_the_archived_trainer(warmup_updates, resample_every, expected):
    # The archived trainer's own cases: fresh draws at 0, R, 2R, ... during warm-up, every update after.
    assert [u for u in range(6) if pino._should_resample_online_batch(u, warmup_updates, resample_every)] == expected


def test_physics_warmup_holds_batches_then_resamples_and_resumes_inside_a_hold(homogeneous_files, tmp_path, monkeypatch):
    inputs, data, norm, gates = homogeneous_files
    cfg = space_time_config()
    cfg['physics_test']['pino'].update(physics_warmup_updates=3, physics_warmup_resample_every=2)
    draws = []
    draw = pino.draw_online_batch
    def record(inputs, generators, count, assignments=None):
        if count == cfg['physics_test']['pino']['batch_size']:
            draws.append(count)
        return draw(inputs, generators, count, assignments)
    monkeypatch.setattr(pino, 'draw_online_batch', record)
    baseline, restarted = tmp_path / 'baseline', tmp_path / 'restart'
    def run(output):
        return pino.run_screen(cfg, data, output, gates[1], 'online', None, norm, smoke=True)
    assert run(baseline)['status'] == 'complete'
    # Update 1 draws, 2 reuses it, 3 draws, 4 is past the warm-up and draws.
    assert len(draws) == 3
    manifest = [json.loads(line) for line in (baseline / 'mg/training_manifest.jsonl').read_text().splitlines()]
    assert [row['resampled'] for row in manifest] == [True, False, True, True]
    assert manifest[1]['problems'] == manifest[0]['problems']
    assert len({json.dumps(row['problems'], sort_keys=True) for row in manifest}) == 3
    original = pino.space_time_backward
    def interrupt(*args):
        value = original(*args)
        signal.getsignal(getattr(signal, 'SIGUSR1', signal.SIGINT))(getattr(signal, 'SIGUSR1', signal.SIGINT), None)
        return value
    monkeypatch.setattr(pino, 'space_time_backward', interrupt)
    assert run(restarted)['status'] == 'interrupted'
    monkeypatch.setattr(pino, 'space_time_backward', original)
    held = torch.load(restarted / 'mg/checkpoint_latest.pt', weights_only=False)
    assert held['successful_updates'] == 1 and held['online_state']['held_problems'] is not None
    assert pino.resume_screen(restarted)['status'] == 'complete'
    a, b = (torch.load(output / 'mg/checkpoint_final.pt', weights_only=False) for output in (baseline, restarted))
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'sampler_rng_states'):
        assert_same_state(a[key], b[key])
    assert (baseline / 'mg/training_manifest.jsonl').read_bytes() == (restarted / 'mg/training_manifest.jsonl').read_bytes()


def test_physics_warmup_is_rejected_where_nothing_is_resampled(homogeneous_files, tmp_path):
    _, data, norm, gates = homogeneous_files
    cfg = config(residual_mode='autoregressive')
    cfg['benchmark']['name'] = 'forcing_homogeneous'
    cfg['physics_test']['pino']['physics_warmup_updates'] = 10
    with pytest.raises(ValueError, match='physics_warmup_updates holds online'):
        pino.run_screen(cfg, data, tmp_path / 'fixed', gates[0], normalization_path=norm, smoke=True)
    assert not (tmp_path / 'fixed').exists()


@pytest.mark.parametrize('objective', ['raw', 'exact', 'mg'])
@pytest.mark.parametrize('model_type,optimizer', [('cvit', 'SOAP'), ('fno', 'SOAP'), ('fno', 'SSESOAP')])
def test_formulation_comparison_matches_architecture_streams_and_budgets(homogeneous_files, tmp_path, objective, model_type, optimizer):
    _, data, norm, _ = homogeneous_files
    cfg = online_config(model_type)
    cfg['training']['optimizer'] = optimizer
    cfg['physics_test']['pino'].update(formulation_comparison=True, controls=[objective])
    outputs = []
    for mode in ('autoregressive', 'time_conditioned'):
        cfg['physics_test']['pino']['residual_mode'] = mode
        output = tmp_path / mode
        result = pino.run_screen(cfg, data, output, stage='online', normalization_path=norm)
        assert result['status'] == 'complete'
        assert result['sampled_cases'] == 4
        assert result['case_interval_residual_evaluations'] == 8
        assert result['formulation_comparison'] and not result['accuracy_gate_applied']
        frozen = yaml.safe_load((output / 'config_used.yaml').read_text())
        protocol = frozen['physics_test']['protocol']
        assert protocol['prefix_horizon'] == .01
        if model_type == 'cvit':
            assert protocol['resolved_cvit_kwargs']['use_time_query']
        else:
            assert protocol['resolved_cvit_kwargs'] is None
            assert protocol['resolved_fno_dims']['cond_static_dim'] == 1
            assert 'FNO_tokens' in protocol['input_encoding']
        assert protocol['prerequisites'] == []
        rows = json.loads((output / objective / 'val_pairs_000002.json').read_text())
        assert {r['time'] for r in rows} == {0, .005, .01, .015, .02}
        assert all(r['rmse_K'] == 0 for r in rows if r['time'] == 0)
        assert all(r['right_wall_rmse_K'] == 0 for r in rows)
        checkpoint = torch.load(output / objective / 'checkpoint_final.pt', weights_only=False)
        assert checkpoint['scheduler_state_dict']['next_update'] == 2
        assert checkpoint['optimizer_state_dict']['state']
        if optimizer == 'SSESOAP':
            source = 'src/operators/ss_esoap.py'
            assert source in protocol['source_sha256']
            assert (output / 'source_snapshot' / source).is_file()
        assert np.isfinite(result['controls'][objective]['full_trajectory_rmse_K'])
        outputs.append(output)
    a, b = outputs
    assert_same_state(torch.load(a / 'initial_state.pt', weights_only=True),
                      torch.load(b / 'initial_state.pt', weights_only=True))
    assert_same_state(np.load(a / 'validation_inputs.npy', allow_pickle=True).tolist(),
                      np.load(b / 'validation_inputs.npy', allow_pickle=True).tolist())
    np.testing.assert_array_equal(np.load(a / 'validation_references.npy'), np.load(b / 'validation_references.npy'))
    assert (a / objective / 'training_manifest.jsonl').read_text() == (b / objective / 'training_manifest.jsonl').read_text()


@pytest.mark.parametrize('interruption', ['training', 'validation'])
@pytest.mark.parametrize('model_type,optimizer', [('cvit', 'SOAP'), ('fno', 'SOAP'), ('fno', 'SSESOAP')])
@pytest.mark.parametrize('detach_previous', [True, False])
def test_time_conditioned_comparison_resume(homogeneous_files, tmp_path, monkeypatch, interruption, model_type, detach_previous, optimizer):
    clock_ticks = count()
    monkeypatch.setattr(pino.time, 'perf_counter', lambda: next(clock_ticks))
    _, data, norm, _ = homogeneous_files
    cfg = online_config(model_type)
    cfg['training']['optimizer'] = optimizer
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned')
    cfg['physics_test']['pino']['detach_previous_predictions'] = detach_previous
    def run(output):
        return pino.run_screen(cfg, data, output, stage='online', normalization_path=norm)
    baseline, resumed = tmp_path / 'baseline', tmp_path / 'resumed'
    assert run(baseline)['status'] == 'complete'
    name = 'time_conditioned_backward' if interruption == 'training' else 'decode_evolution'
    original = getattr(pino, name)
    calls = []
    def interrupt(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 1:
            stop_signal = getattr(signal, 'SIGUSR1', signal.SIGINT)
            signal.getsignal(stop_signal)(stop_signal, None)
        return result
    monkeypatch.setattr(pino, name, interrupt)
    assert run(resumed)['status'] == 'interrupted'
    monkeypatch.setattr(pino, name, original)
    assert pino.resume_screen(resumed)['status'] == 'complete'
    a, b = load_checkpoint(baseline), load_checkpoint(resumed)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'sampler_rng_states'):
        assert_same_state(a[key], b[key])
    assert_same_state(json.loads((baseline / 'mg/val_pairs_000002.json').read_text()),
                      json.loads((resumed / 'mg/val_pairs_000002.json').read_text()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize('optimizer', ['SSESOAP'])
def test_structured_optimizer_time_conditioned_fno_cuda_smoke(homogeneous_files, tmp_path, optimizer):
    _, data, norm, _ = homogeneous_files
    cfg = online_config('fno')
    cfg['training'].update(optimizer=optimizer, device='cuda')
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned',
        detach_previous_predictions=False, space_time_preconditioning=True)
    output = tmp_path / 'cuda'
    result = pino.run_screen(cfg, data, output, stage='online', normalization_path=norm)
    assert result['status'] == 'complete'
    checkpoint = load_checkpoint(output)
    momentum = 'momentum'
    factors = 'bases'
    assert all(state['step'] == 2 for state in checkpoint['optimizer_state_dict']['state'].values())
    spectral = [state for state in checkpoint['optimizer_state_dict']['state'].values()
                if state[momentum].ndim == 5]
    assert spectral and all(len(state[factors]) == 5 for state in spectral)
    assert np.isfinite(result['controls']['mg']['full_trajectory_rmse_K'])


@pytest.mark.parametrize('model_type', ['cvit', 'fno'])
@pytest.mark.parametrize('detach_previous', [True, False])
@pytest.mark.parametrize('objective', ['raw', 'exact', 'mg'])
def test_time_conditioned_gradient_matches_trajectory_objective(model_type, detach_previous, objective):
    inputs, _ = prescribed_inputs()
    cfg = online_config(model_type)
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned')
    cfg['physics_test']['pino']['detach_previous_predictions'] = detach_previous
    model = pino.build_screen_model(cfg, inputs, torch.device('cpu'))
    reference = copy.deepcopy(model)
    params = list(inputs['params'][:2])
    x, y = torch.meshgrid(torch.tensor(inputs['x'], dtype=torch.float32),
                          torch.tensor(inputs['y'], dtype=torch.float32), indexing='ij')
    coords = torch.stack((x, y), -1).reshape(1, -1, 2)
    geom = build_homogeneous_cn_geom(inputs['x'], inputs['y'], 1, inputs['dt'], sigma_global=10)
    inverse = (pino.MGOneCycleInverse(geom, coarse_max_unknowns=4) if objective == 'mg'
               else pino.ExactCNInverse(geom) if objective == 'exact' else None)
    loss, _, _ = pino.time_conditioned_backward(model, params, cfg, inputs, coords, geom, inverse, objective)
    forcing, initial = pino.model_inputs(params, cfg, inputs, coords.device)
    latent = forcing if model_type == 'fno' else reference.encode(forcing, initial[:, None])
    states = [initial] + [pino.decode_evolution(reference, latent, initial, coords, n * inputs['dt'], cfg, inputs)
                          for n in (1, 2)]
    residuals = [cn_step_residual(states[n], states[n + 1], geom,
                    pino.make_boundary(params, [n, n], inputs, coords.device, torch.float32),
                    detach_previous=detach_previous) for n in (0, 1)]
    losses = [(inverse(r) if inverse is not None else r).square().mean() for r in residuals]
    expected = torch.stack(losses).mean()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-6, rtol=1e-5)
    branches = (model.temporal_encoder, model.cond_mlp, model.spectral_layers) if model_type == 'fno' else (
        model.forcing_encoder, model.decoder.time_film)
    for branch in branches:
        assert sum(p.grad.abs().sum() for p in branch.parameters() if p.grad is not None) > 0
    latent = forcing if model_type == 'fno' else model.encode(forcing, initial[:, None])
    zero = pino.decode_evolution(model, latent, initial, coords, 0., cfg, inputs)
    torch.testing.assert_close(zero, initial, atol=0, rtol=0)
    arbitrary = pino.decode_evolution(model, latent, initial, coords, .0073, cfg, inputs)
    assert torch.isfinite(arbitrary).all()
    torch.testing.assert_close(arbitrary[:, -1], initial[:, -1], atol=0, rtol=0)


@pytest.mark.parametrize('objective', ['raw', 'exact', 'mg'])
def test_cn_objective_can_differentiate_previous_prediction(objective):
    inputs, _ = prescribed_inputs()
    geom = build_homogeneous_cn_geom(inputs['x'], inputs['y'], 1, inputs['dt'], sigma_global=10)
    inverse = (pino.MGOneCycleInverse(geom, coarse_max_unknowns=4) if objective == 'mg'
               else pino.ExactCNInverse(geom) if objective == 'exact' else None)
    previous = torch.randn(2, 4, 4, requires_grad=True)
    following = torch.randn(2, 4, 4, requires_grad=True)
    bc = pino.make_boundary(inputs['params'][:2], [1, 1], inputs, torch.device('cpu'), torch.float32)
    detached = pino.cn_objective(previous, following, geom, bc, inverse, objective)[0]
    full = pino.cn_objective(previous, following, geom, bc, inverse, objective, detach_previous=False)[0]
    torch.testing.assert_close(detached, full, atol=0, rtol=0)
    old_previous, old_following = torch.autograd.grad(detached, (previous, following), allow_unused=True)
    new_previous, new_following = torch.autograd.grad(full, (previous, following))
    assert old_previous is None
    assert torch.isfinite(new_previous).all() and new_previous.abs().sum() > 0
    torch.testing.assert_close(old_following, new_following, atol=0, rtol=0)


@pytest.mark.parametrize('objective', ['exact', 'mg'])
@pytest.mark.parametrize('detach_previous', [False, True])
def test_time_conditioned_space_time_checkpointed_gradient_matches_full_graph(objective, detach_previous):
    inputs, _ = prescribed_inputs(n=8)
    cfg = online_config('fno')
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned',
        detach_previous_predictions=detach_previous, space_time_preconditioning=True, prefix_steps=4)
    model = pino.build_screen_model(cfg, inputs, torch.device('cpu'))
    reference = copy.deepcopy(model)
    params = list(inputs['params'][:2])
    x, y = torch.meshgrid(torch.tensor(inputs['x'], dtype=torch.float32),
                          torch.tensor(inputs['y'], dtype=torch.float32), indexing='ij')
    coords = torch.stack((x, y), -1).reshape(1, -1, 2)
    geom = build_homogeneous_cn_geom(inputs['x'], inputs['y'], 1, inputs['dt'], sigma_global=10)
    inverse = (pino.MGOneCycleInverse(geom, coarse_max_unknowns=4) if objective == 'mg'
               else pino.ExactCNInverse(geom))
    actual, residual, transformed = pino.time_conditioned_backward(
        model, params, cfg, inputs, coords, geom, inverse, objective)
    latent, initial = pino.model_inputs(params, cfg, inputs, coords.device)
    states = torch.stack([initial] + [pino.decode_evolution(reference, latent, initial, coords,
        n * inputs['dt'], cfg, inputs) for n in range(1, 5)], dim=1)
    boundaries = [pino.make_boundary(params, [n, n], inputs, coords.device, torch.float32) for n in range(4)]
    expected, expected_residual, expected_transformed = pino.prefix_cn_objective(
        states, geom, boundaries, inverse, objective, detach_previous=detach_previous)
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(transformed, expected_transformed)
    full_values = pino.prefix_cn_objective(states, geom, boundaries, inverse, objective)
    for value, full_value in zip((actual, residual, transformed), full_values):
        torch.testing.assert_close(value, full_value, rtol=0, atol=0)
    for value, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(value.grad, wanted.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('detach_previous', [False, True])
@pytest.mark.parametrize('optimizer', ['SOAP', 'SSESOAP'])
def test_time_conditioned_space_time_resume_preserves_full_gradient_protocol(homogeneous_files, tmp_path, monkeypatch, detach_previous, optimizer):
    _, data, norm, _ = homogeneous_files
    cfg = online_config('fno')
    cfg['training']['optimizer'] = optimizer
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned',
        detach_previous_predictions=detach_previous, space_time_preconditioning=True)
    baseline, interrupted = tmp_path / 'baseline', tmp_path / 'interrupted'
    assert pino.run_screen(cfg, data, baseline, stage='online', normalization_path=norm)['status'] == 'complete'
    original = pino.time_conditioned_backward

    def stop_after_update(*args):
        result = original(*args)
        stop_signal = getattr(signal, 'SIGUSR1', signal.SIGINT)
        signal.getsignal(stop_signal)(stop_signal, None)
        return result

    monkeypatch.setattr(pino, 'time_conditioned_backward', stop_after_update)
    assert pino.run_screen(cfg, data, interrupted, stage='online', normalization_path=norm)['status'] == 'interrupted'
    monkeypatch.setattr(pino, 'time_conditioned_backward', original)
    assert pino.resume_screen(interrupted)['status'] == 'complete'
    expected, actual = load_checkpoint(baseline), load_checkpoint(interrupted)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'sampler_rng_states'):
        assert_same_state(actual[key], expected[key])
    protocol = actual['config']['physics_test']['protocol']
    assert protocol['activation_checkpointing']
    assert protocol['temporal_gradient_policy'] == (
        'detached_previous_prediction_and_correction_with_activation_recomputation' if detach_previous
        else 'full_space_time_graph_with_activation_recomputation')
    assert protocol['preconditioner'] == 'space_time_MG_V_cycle_forward_substitution'


def test_space_time_screen_can_pause_after_validation_without_an_extra_update(homogeneous_files, tmp_path, monkeypatch):
    _, data, norm, _ = homogeneous_files
    cfg = online_config('fno')
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned',
        detach_previous_predictions=True, space_time_preconditioning=True,
        online_validate_every=1, full_validate_every=1)
    evaluate = pino.evaluate_online_mg
    calls = count()

    def pause_after_first_trained_validation(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if next(calls) == 1:
            stop_signal = getattr(signal, 'SIGUSR1', signal.SIGINT)
            signal.getsignal(stop_signal)(stop_signal, None)
        return result

    monkeypatch.setattr(pino, 'evaluate_online_mg', pause_after_first_trained_validation)
    output = tmp_path / 'screen'
    result = pino.run_screen(cfg, data, output, stage='online', normalization_path=norm)
    assert result['status'] == 'interrupted' and result['successful_updates'] == 1
    checkpoint = load_checkpoint(output, 'checkpoint_latest.pt')
    assert not checkpoint['online_state']['pending_validation']
    assert (output / 'mg/validation_000001.json').exists()
    monkeypatch.setattr(pino, 'evaluate_online_mg', evaluate)
    assert pino.resume_screen(output)['successful_updates'] == 2


def test_fno_comparison_requires_its_own_hard_wall(homogeneous_files, tmp_path):
    _, data, norm, _ = homogeneous_files
    cfg = online_config('fno')
    cfg['physics_test']['pino'].update(formulation_comparison=True, residual_mode='time_conditioned')
    cfg['model']['parameters']['hard_right_dirichlet'] = False
    with pytest.raises(ValueError, match='hard_right_dirichlet=true'):
        pino.run_screen(cfg, data, tmp_path / 'invalid', stage='online', normalization_path=norm)
