import copy
import csv
import json
import os
from pathlib import Path
import signal
import subprocess

import numpy as np
import pytest
import torch
import yaml

import src.operators.train_pino as pino
from scripts.run_train_pino import result_exit_code
from scripts.run_train_fixed import _parse_override_value
from src.operators.cvit import ForcingICCViT
from src.physics.fv_preconditioner import MGOneCycleInverse
from src.physics.init_conditions import IC_FAMILIES
from tests.test_cvit_pino import config, prescribed_inputs, save_inputs


def online_config(updates=4):
    cfg = config(residual_mode='mg')
    cfg['physics_test']['cvit']['fourier_freq'] = 20
    cfg['physics_test']['pino'].update(online_sampling='per_update', compact_metrics=True,
        online_updates=updates, online_validate_every=2, full_validate_every=4,
        prefix_steps=10, validation_rollout_steps=60, batch_size=8,
        validation_cases=4, validation_batch_size=8, log_every=2,
        startup_profile_updates=[])
    return cfg


@pytest.fixture
def online_inputs(tmp_path):
    inputs, _ = prescribed_inputs()
    inputs['t_final'] = .3
    data = tmp_path / 'data'
    data.mkdir()
    normalization = save_inputs(data, inputs)
    np.save(data / 't_grid.npy', [0, .005, .3])
    gates = []
    for stage in ('phase4_direct_state_mg', 'phase4_fixed_cvit_autoregressive_prefix_mg'):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / 'final_metrics.json').write_text(json.dumps(dict(stage=stage, passed=True)))
        gates.append(directory)
    return inputs, data, normalization, gates


def run_online(cfg, fixture, output):
    _, data, normalization, gates = fixture
    return pino.run_screen(cfg, data, output, gates[0], 'online', gates[1], normalization, smoke=True)


def load_checkpoint(output, name='checkpoint_final.pt'):
    return torch.load(output / 'mg' / name, map_location='cpu', weights_only=False)


def assert_same_state(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same_state(x, y)
    else:
        assert a == b


def test_online_samples_are_fresh_reproducible_and_independent(online_inputs):
    inputs = online_inputs[0]
    a, b = pino.online_generators(42), pino.online_generators(42)
    first = pino.draw_online_batch(inputs, a, 8)
    assert_same_state(first, pino.draw_online_batch(inputs, b, 8))
    second = pino.draw_online_batch(inputs, a, 8)
    assert not np.array_equal(first[0]['T0'], second[0]['T0'])
    assert first[0]['temporal_params'] != second[0]['temporal_params']
    independent = pino.online_generators(42)
    independent['ic_params'].random(1000)
    changed_ic = pino.draw_online_batch(inputs, independent, 8)
    assert [p['temporal_params'] for p in first] == [p['temporal_params'] for p in changed_ic]
    cfg = online_config()
    cfg['physics_test']['pino']['validation_cases'] = 128
    frozen = pino.matched_stream(cfg, inputs, 'online')
    assert_same_state(frozen, pino.matched_stream(cfg, inputs, 'online'))
    assert frozen['problems'] == [] and frozen['ids'].size == 0
    assert {f: sum(p['ic_family'] == f for p in frozen['validation']) for f in IC_FAMILIES} == dict.fromkeys(IC_FAMILIES, 32)
    training_keys = {json.dumps(p['temporal_params'], sort_keys=True) for p in first + second}
    assert not training_keys & {json.dumps(p['temporal_params'], sort_keys=True) for p in frozen['validation']}


def test_rollout_sampling_mg_updates_and_six_field_window(online_inputs, tmp_path, monkeypatch, capsys):
    cfg, output = online_config(2), tmp_path / 'run'
    cfg['physics_test']['pino'].update(online_validate_every=1, full_validate_every=2)
    batches, encoded, decoded, cycles, steps, clipped, losses = [], [], [], [], [], [], []
    draw, encode, decode = pino.draw_online_batch, ForcingICCViT.encode, pino.decode_grid
    cycle, step, clip, backward = MGOneCycleInverse._cycle, torch.optim.Adam.step, torch.nn.utils.clip_grad_norm_, pino.autoregressive_backward
    def capture_draw(inputs, rng, count, families=None):
        result = draw(inputs, rng, count, families)
        if count == 8:
            batches.append(copy.deepcopy(result))
        return result
    def capture_encode(self, forcing, state):
        if torch.is_grad_enabled():
            assert not state.requires_grad
            encoded.append(state[:, 0].clone())
        return encode(self, forcing, state)
    def capture_decode(*args):
        result = decode(*args)
        if torch.is_grad_enabled():
            decoded.append(result.detach().clone())
        return result
    def capture_cycle(self, rhs, level):
        if level == 0:
            cycles.append(tuple(rhs.shape))
        return cycle(self, rhs, level)
    def capture_step(self, *args, **kwargs):
        steps.append(self.param_groups[0]['lr'])
        return step(self, *args, **kwargs)
    def capture_clip(*args, **kwargs):
        result = clip(*args, **kwargs)
        clipped.append(float(result))
        return result
    def capture_backward(*args):
        loss, residual, correction = backward(*args)
        losses.append((float(loss), float(residual.square().mean())))
        return loss, residual, correction
    monkeypatch.setattr(pino, 'draw_online_batch', capture_draw)
    monkeypatch.setattr(ForcingICCViT, 'encode', capture_encode)
    monkeypatch.setattr(pino, 'decode_grid', capture_decode)
    monkeypatch.setattr(MGOneCycleInverse, '_cycle', capture_cycle)
    monkeypatch.setattr(torch.optim.Adam, 'step', capture_step)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture_clip)
    monkeypatch.setattr(pino, 'autoregressive_backward', capture_backward)
    monkeypatch.setattr(pino, 'ExactCNInverse', lambda *args: pytest.fail('Online MG must not use exact inverse'))
    result = run_online(cfg, online_inputs, output)
    assert result['status'] == 'complete' and not result['accuracy_gate_applied']
    assert len(batches) == len(steps) == len(clipped) == 2
    assert len(encoded) == len(decoded) == len(cycles) == 20
    assert all(shape[0] == 8 for shape in cycles)
    for u in range(2):
        _, ic = pino.model_inputs(batches[u], cfg, online_inputs[0], torch.device('cpu'))
        torch.testing.assert_close(encoded[u * 10], ic, rtol=0, atol=0)
        for n in range(1, 10):
            torch.testing.assert_close(encoded[u * 10 + n], decoded[u * 10 + n - 1], rtol=0, atol=0)
    assert batches[0][0]['temporal_params'] != batches[1][0]['temporal_params']
    with (output / 'mg/train_metrics.csv').open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ['update', 'physics_loss', 'raw_residual_rms', 'gradient_norm', 'learning_rate', 'seconds_per_update']
        rows = list(reader)
    assert len(rows) == 1
    assert float(rows[0]['physics_loss']) == pytest.approx(np.mean([v[0] for v in losses]))
    assert float(rows[0]['raw_residual_rms']) == pytest.approx(np.sqrt(np.mean([v[1] for v in losses])))
    assert float(rows[0]['gradient_norm']) == pytest.approx(np.mean(clipped))
    assert float(rows[0]['learning_rate']) == steps[-1]
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if line.startswith('train ')][0].split()[1:]) == 6
    validation = [line for line in lines if line.startswith('validation ')]
    assert len(validation) == 3
    assert 'first_step_sigma_percent=' in validation[0]
    assert 'full_rmse_K=' not in validation[1]
    assert {path.name for path in (output / 'mg').glob('checkpoint_*.pt')} == {'checkpoint_000000.pt', 'checkpoint_latest.pt', 'checkpoint_final.pt'}
    manifest = [json.loads(line) for line in (output / 'mg/training_manifest.jsonl').read_text().splitlines()]
    assert all(len(row['problems']) == 8 and all('T0' not in p for p in row['problems']) for row in manifest)


def test_validation_cadence_and_independent_rmse(online_inputs, tmp_path, monkeypatch):
    cfg = online_config(10000)
    cfg['physics_test']['pino'].update(online_validate_every=500, full_validate_every=1000)
    expected = [60 if u % 1000 == 0 else 10 for u in range(0, 10001, 500)]
    assert [pino.validation_steps(u, cfg) for u in range(0, 10001, 500)] == expected
    assert all(pino.validation_steps(u, cfg) == 60 for u in [0, 5000, 10000])
    cfg = online_config(2)
    output = tmp_path / 'run'
    predictions = []
    decode = pino.decode_grid
    def capture_decode(*args):
        result = decode(*args)
        if not torch.is_grad_enabled():
            predictions.append(result.double().numpy() * 10 + 295)
        return result
    monkeypatch.setattr(pino, 'decode_grid', capture_decode)
    run_online(cfg, online_inputs, output)
    refs = np.load(output / 'validation_references.npy')
    physical = np.stack(predictions[:60], axis=1)
    error = physical - refs[:, 1:]
    summary = json.loads((output / 'mg/validation_000000.json').read_text())
    for key, subset in [('first_step_rmse_K', error[:, :1]), ('prefix_trajectory_rmse_K', error[:, :10]), ('full_trajectory_rmse_K', error)]:
        assert summary[key] == pytest.approx(np.sqrt(np.mean(subset ** 2)), rel=1e-12)
        assert summary[key.replace('rmse_K', 'sigma_error_percent')] == pytest.approx(summary[key] * 10)
    rows = json.loads((output / 'mg/val_pairs_000000.json').read_text())
    assert len(rows) == 4 * 61
    assert all(len([r for r in rows if r['ic_family'] == family]) == 61 for family in IC_FAMILIES)
    frozen = yaml.safe_load((output / 'config_used.yaml').read_text())
    for name, digest in frozen['physics_test']['protocol']['stream_sha256'].items():
        assert pino.file_sha256(output / name) == digest
    assert frozen['physics_test']['protocol']['reference_role'] == 'held_out_validation_only'


@pytest.mark.parametrize('interruption', ['training', 'validation', 'prefix_validation', 'validation_batch_boundary'])
def test_signal_resume_matches_uninterrupted_and_reuses_reference_cache(interruption, online_inputs, tmp_path, monkeypatch):
    cfg = online_config(4)
    if interruption == 'validation_batch_boundary':
        cfg['physics_test']['pino']['validation_batch_size'] = 2
    baseline, restarted = tmp_path / 'baseline', tmp_path / 'restarted'
    run_online(cfg, online_inputs, baseline)
    triggered, validation_calls = [], []
    original = pino.autoregressive_backward if interruption == 'training' else pino.decode_grid
    def interrupt(*args):
        result = original(*args)
        if not torch.is_grad_enabled():
            validation_calls.append(True)
        target_calls = 62 if interruption == 'prefix_validation' else 60 if interruption == 'validation_batch_boundary' else 1
        if not triggered and (interruption == 'training' or (not torch.is_grad_enabled() and len(validation_calls) == target_calls)):
            triggered.append(True)
            os.kill(os.getpid(), signal.SIGUSR1)
        return result
    name = 'autoregressive_backward' if interruption == 'training' else 'decode_grid'
    prior_handler = signal.getsignal(signal.SIGUSR1)
    monkeypatch.setattr(pino, name, interrupt)
    result = run_online(cfg, online_inputs, restarted)
    assert result['status'] == 'interrupted' and result_exit_code(result) == 75
    assert signal.getsignal(signal.SIGUSR1) == prior_handler
    checkpoint = load_checkpoint(restarted, 'checkpoint_latest.pt')
    if interruption == 'training':
        assert checkpoint['successful_updates'] == 1
        assert len(checkpoint['online_state']['window']) == 1
    else:
        progress = checkpoint['online_state']['validation_progress']
        assert checkpoint['successful_updates'] == (2 if interruption == 'prefix_validation' else 0)
        if interruption == 'validation_batch_boundary':
            assert progress['case_start'] == 2 and progress['next_step'] == 0 and progress['state'] is None
        else:
            assert progress['next_step'] == (3 if interruption == 'prefix_validation' else 2)
            assert progress['state'].device.type == 'cpu'
    for filename in ['training_manifest.jsonl', 'train_metrics.csv']:
        with (restarted / 'mg' / filename).open('a') as file:
            file.write('uncheckpointed trailing data\n')
    monkeypatch.setattr(pino, name, original)
    reference_hash = pino.file_sha256(restarted / 'validation_references.npy')
    monkeypatch.setattr(pino, 'development_references', lambda *args: pytest.fail('Resume must reuse FV cache'))
    result = pino.resume_screen(restarted)
    assert result['status'] == 'complete' and result_exit_code(result) == 0
    assert pino.file_sha256(restarted / 'validation_references.npy') == reference_hash
    a, b = load_checkpoint(baseline), load_checkpoint(restarted)
    for key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'sampler_rng_states', 'torch_rng_state', 'numpy_rng_state', 'cuda_rng_states']:
        assert_same_state(a[key], b[key])
    assert (baseline / 'mg/training_manifest.jsonl').read_bytes() == (restarted / 'mg/training_manifest.jsonl').read_bytes()
    for u in [0, 2, 4]:
        assert_same_state(json.loads((baseline / f'mg/val_pairs_{u:06d}.json').read_text()),
                          json.loads((restarted / f'mg/val_pairs_{u:06d}.json').read_text()))
    with (restarted / 'mg/train_metrics.csv').open() as file:
        assert [int(r['update']) for r in csv.DictReader(file)] == [2, 4]


def test_prerequisites_and_checkpoint_before_each_validation(online_inputs, tmp_path, monkeypatch):
    cfg, output = online_config(2), tmp_path / 'run'
    cfg['physics_test']['pino'].update(online_validate_every=1, full_validate_every=2)
    evaluator = pino.evaluate_online_mg
    checkpoints = []
    def audit(model, config, *args):
        saved = load_checkpoint(output, 'checkpoint_latest.pt')
        update = saved['successful_updates']
        checkpoints.append(update)
        assert saved['online_state']['pending_validation']
        assert_same_state(saved['model_state_dict'], model.state_dict())
        if update == 2:
            assert_same_state(load_checkpoint(output)['model_state_dict'], model.state_dict())
        return evaluator(model, config, *args)
    monkeypatch.setattr(pino, 'evaluate_online_mg', audit)
    run_online(cfg, online_inputs, output)
    assert checkpoints == [0, 1, 2]
    gate = online_inputs[3][1] / 'final_metrics.json'
    gate.write_text(json.dumps(dict(stage='phase4_fixed_cvit_autoregressive_prefix_mg', passed=False)))
    with pytest.raises(ValueError, match='did not pass'):
        run_online(cfg, online_inputs, tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()


def test_throughput_windows_and_remaining_validation_cost():
    cfg = online_config(10000)['physics_test']['pino']
    cfg.update(online_validate_every=500, full_validate_every=1000)
    durations = [(u, 100 if u <= 10 else 2 if u <= 50 else 3) for u in range(1, 101)]
    a = pino.runtime_report(50, durations, dict(prefix=4, full=20), cfg, 2 ** 30, 21000)
    b = pino.runtime_report(100, durations, dict(prefix=4, full=20), cfg, None, 20000)
    assert a['mean_seconds_per_update'] == 2 and a['updates_per_second'] == .5
    assert b['mean_seconds_per_update'] == 3 and b['updates_per_second'] == 1 / 3
    assert a['estimated_remaining_total_seconds'] == (10000 - 50) * 2 + 10 * 4 + 10 * 20
    assert a['estimated_to_fit_allocation'] and not b['estimated_to_fit_allocation']
    assert a['peak_GPU_memory_GiB'] == 1 and b['peak_GPU_memory_GiB'] is None


def test_runtime_diagnostics_saved_at_50_and_100(online_inputs, tmp_path, monkeypatch):
    cfg = online_config(100)
    cfg['physics_test']['pino'].update(online_validate_every=500, full_validate_every=1000,
                                    log_every=50, startup_profile_updates=[50, 100])
    def cheap_update(model, *args):
        loss = next(model.parameters()).square().mean()
        loss.backward()
        return loss.detach(), torch.ones(10, 8, 3, 4), torch.ones(10, 8, 3, 4)
    monkeypatch.setattr(pino, 'autoregressive_backward', cheap_update)
    output = tmp_path / 'run'
    run_online(cfg, online_inputs, output)
    reports = json.loads((output / 'mg/runtime_diagnostics.json').read_text())
    assert [r['update'] for r in reports] == [50, 100]
    durations = load_checkpoint(output)['online_state']['profile_durations']
    assert reports[0]['mean_seconds_per_update'] == pytest.approx(np.mean([t for u, t in durations if 10 < u <= 50]))
    assert reports[1]['mean_seconds_per_update'] == pytest.approx(np.mean([t for u, t in durations if 50 < u <= 100]))
    assert all(r['validation_wall_seconds']['full'] > r['validation_wall_seconds']['prefix'] > 0 for r in reports)


def test_launcher_resources_overrides_and_resume(tmp_path):
    script = Path('slurm/train_pcvit_mg_msi.sbatch').resolve()
    subprocess.run(['bash', '-n', script], check=True)
    text = script.read_text()
    for directive in ['--partition=msigpu', '--gres=gpu:a100:1', '--time=06:00:00', '--cpus-per-task=4', '--mem=32G', '--signal=USR1@120']:
        assert directive in text
    project = tmp_path / 'project with spaces'
    project.mkdir()
    capture = tmp_path / 'arguments.json'
    prologue = '''module() { :; }
conda() { :; }
nvidia-smi() { :; }
srun() { printf '%s\\0' "$@" > "$CAPTURE"; }
export -f module conda nvidia-smi srun
exec bash "$LAUNCHER"
'''
    env = dict(os.environ, SLURM_JOB_ID='123', SLURM_SUBMIT_DIR=str(project),
               PERSISTENT_RUNS=str(tmp_path / 'persistent'), CAPTURE=str(capture), LAUNCHER=str(script))
    env.pop('PROJECT_DIR', None)
    env.pop('RESUME_RUN_DIR', None)
    subprocess.run(['bash', '-c', prologue], env=env, check=True, capture_output=True)
    args = capture.read_bytes().decode().strip('\0').split('\0')
    assert args[:3] == ['python', '-u', 'scripts/run_train_pino.py']
    assert args[args.index('--data-dir') + 1] == str(project / 'data/physics_test_single_development_20261001')
    overrides = dict(arg.split('=', 1) for arg in args if '=' in arg)
    assert overrides['physics_test.pino.online_updates'] == '10000'
    assert overrides['physics_test.pino.validation_cases'] == '128'
    assert overrides['physics_test.pino.validation_batch_size'] == '8'
    assert overrides['physics_test.pino.prefix_steps'] == '10'
    assert overrides['physics_test.cvit.fourier_freq'] == '20'
    assert _parse_override_value(overrides['physics_test.pino.controls']) == ['mg']
    env['RESUME_RUN_DIR'] = str(tmp_path / 'resume with spaces')
    subprocess.run(['bash', '-c', prologue], env=env, check=True, capture_output=True)
    args = capture.read_bytes().decode().strip('\0').split('\0')
    assert args[3:] == ['--resume', '--output-dir', env['RESUME_RUN_DIR']]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_online_mg_smoke(online_inputs, tmp_path):
    cfg = online_config(1)
    cfg['training']['device'] = 'cuda'
    result = run_online(cfg, online_inputs, tmp_path / 'cuda')
    assert result['status'] == 'complete' and result['successful_updates'] == 1
