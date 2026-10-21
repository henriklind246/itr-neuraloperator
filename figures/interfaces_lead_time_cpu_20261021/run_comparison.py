"""Lead-time sweep of paired CPU cost and FNO accuracy for fresh B2 cases.

B2 is the ``interfaces`` benchmark: varying interface location, scalar ITR and
varying initial condition. For every lead time on the 0.01 s snapshot grid this
measures, against the FV reference at the B2 training timestep:

- the single-thread CPU cost of advancing the FV solver over the lead,
- the single-thread CPU cost of one FNO prediction over the same interval,
- their ratio, and
- the FNO field error over every (t_s, t_j) pair with that lead.

    python figures/interfaces_lead_time_cpu_20261021/run_comparison.py \
        --checkpoint <run>/config0/seed32/fno2d_best.pt

The error sweep is batched on ``--device`` (CUDA when present); it runs after
the CPU timing and does not affect it. ``--plot-only`` redraws the figure from
the CSV already in ``--out``.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits, threadpool_info
from torch.utils.data import DataLoader
from data.generate_dataset import build_base_setup, resolve_seed_streams
from data.dataset import SnapshotPairDataset, collate_fn, problem_from_config, T_EPS
from scripts.invert import load_checkpoint
from src.operators.eval import evaluate

OUT = Path(__file__).resolve().parent
FIGURE_KEY = 'interfaces_lead_time_cpu'
SEED = 20261021
T_FINAL = 0.3
SNAPSHOT_DT = 0.01
# slurm/generate_data_msi.sbatch builds the B2 training set with --dt 0.0003125
# --save-stride 32 --ramp-seconds 0.01: 960 solver steps to t_final.
DT = 0.0003125
RAMP_SECONDS = 0.01
T_REF = 300.0
# Bound on how far the device sweep may move a timed pair's RMSE relative to the
# CPU prediction. Pointwise differences are recorded but not gated: MPS differs
# from CPU by a few mK, which is immaterial at RMSE ~0.1 K.
CPU_DEVICE_RMSE_RTOL = 0.01


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=Path,
                        help='B2 fno2d_best.pt trained on the dt=0.0003125 interfaces data.')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--num-sims', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=5, help='Timing repeats per simulation.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Generator draw family; 0 is the family the checkpoint trained on.')
    parser.add_argument('--dt', type=float, default=DT, help='FV solver timestep [s].')
    parser.add_argument('--device', default='auto', help='Device for the batched error sweep.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--check-sims', type=int, default=4,
                        help='Simulations re-scored with evaluate() to cross-check the metrics.')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    if not args.plot_only and args.checkpoint is None:
        parser.error('--checkpoint is required unless --plot-only is given')
    if args.seed == 0:
        parser.error('--seed 0 is the training draw family; pick a held-out family')
    return args


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def cpu_name():
    if sys.platform == 'darwin':
        return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def draw(spec, count, seed, dt, stride):
    # The sin sampler sees dt, so draws must use the timestep the training set used.
    seeds = resolve_seed_streams(seed)
    setup = build_base_setup(count, stride, t_final=T_FINAL, dt=dt, ramp_seconds=RAMP_SECONDS,
                             lhs_seed=seeds['lhs'], forcing_profile_seed=seeds['forcing_profile'])
    params = spec.sample_sim_params(np.random.default_rng(seeds['ic_params']),
                                   np.random.default_rng(seeds['forcing_profile']),
                                   setup['grids'], setup['time_cfg'])
    return params, setup


def dataset(params, fields, t_grid, setup, spec, loaded):
    return SnapshotPairDataset(
        trajectories=fields, t_grid=np.asarray(t_grid, np.float32),
        x_grid=setup['grids']['x_grid'].astype(np.float32),
        y_grid=setup['grids']['y_grid'].astype(np.float32),
        sim_ids=np.arange(len(params)), sim_params=np.array(params, dtype=object),
        mu_global=loaded.mu_global, sigma_global=loaded.sigma_global,
        dt=setup['dt'], t_final=T_FINAL, ramp_seconds=RAMP_SECONDS, problem=spec,
    )


def fv_trajectory(solver, T0, stride):
    """FVSolver2D.solve, timestamped and stored at every saved snapshot."""
    T = np.asarray(T0, dtype=float)
    t = solver.t
    stamps = np.empty((len(t) - 1) // stride + 1, dtype=np.int64)
    snaps = [T]
    stamps[0] = time.perf_counter_ns()
    for n in range(len(t) - 1):
        T = solver.cn_step(T, t[n])
        if (n + 1) % stride == 0:
            stamps[(n + 1) // stride] = time.perf_counter_ns()
            snaps.append(T)
    return np.stack(snaps), stamps


def training_timestep(config, dt):
    """Check the training set's solver dt and ramp when its directory is reachable."""
    data_dir = Path(config['data']['t_grid_path']).parent
    found = {'training_data_dir': str(data_dir)}
    for key, expected in (('dt', dt), ('ramp_seconds', RAMP_SECONDS)):
        path = data_dir / f'{key}.npy'
        found[key] = float(np.load(path)) if path.is_file() else None
        if found[key] is not None and not np.isclose(found[key], expected, rtol=0.0, atol=1e-12):
            raise SystemExit(f'{path} is {found[key]} but this comparison uses {expected}; the FV '
                             'reference would not match the data the checkpoint trained on.')
    found['verified'] = None not in (found['dt'], found['ramp_seconds'])
    if not found['verified']:
        print(f'WARNING: {data_dir} is not readable here, so the training dt and ramp are '
              'unverified.', flush=True)
    return found


def training_indices(config, n_snap):
    """Snapshot indices the training split drew pairs from (SnapshotPairDataset rule)."""
    n = config['training'].get('n_snapshots')
    if n is None or n >= n_snap:
        return np.arange(n_snap)
    return np.round(np.linspace(0, n_snap - 1, n)).astype(int)


def jsonable(value):
    return value.tolist() if hasattr(value, 'tolist') else str(value)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    # TF32 would make the device sweep disagree with the CPU predictions it is checked against.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    checkpoint = args.checkpoint.expanduser().resolve()
    loaded = load_checkpoint(str(checkpoint), device='cpu')
    model = loaded.model
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert all(torch.equal(v, ckpt['model_state'][k]) for k, v in model.state_dict().items())
    spec = problem_from_config(loaded.config)
    if spec.name != 'interfaces':
        raise SystemExit(f'{checkpoint} is a {spec.name!r} checkpoint, not B2 (interfaces).')

    stride = int(round(SNAPSHOT_DT / args.dt))
    n_steps = int(round(T_FINAL / args.dt))
    if not (np.isclose(stride * args.dt, SNAPSHOT_DT, rtol=0.0, atol=1e-12)
            and np.isclose(n_steps * args.dt, T_FINAL, rtol=0.0, atol=1e-12)):
        raise SystemExit(f'dt={args.dt} must divide both the {SNAPSHOT_DT} s snapshot spacing and t_final.')
    n_snap = n_steps // stride + 1
    t_grid = np.linspace(0.0, T_FINAL, n_snap)
    pop = ckpt['normalization_provenance']['training_population']
    assert len(pop['x_grid']) == len(pop['y_grid']) == 100
    np.testing.assert_allclose(pop['t_grid'], t_grid, rtol=0.0, atol=1e-6)
    training_data = training_timestep(loaded.config, args.dt)
    trained = set(training_indices(loaded.config, n_snap).tolist())
    device = pick_device(args.device)
    print(f'B2 checkpoint {checkpoint.name}: FV dt={args.dt} s, {n_steps} steps, {n_snap} snapshots; '
          f'error sweep on {device}.', flush=True)

    params, setup = draw(spec, args.num_sims, args.seed, args.dt, stride)
    warm_params, warm_setup = draw(spec, 1, args.seed + 1, args.dt, stride)
    N, R = len(params), args.repeats
    leads = np.arange(1, n_snap)
    L = leads.size
    Nx, Ny = setup['Nx'], setup['Ny']
    mu, sigma = loaded.mu_global, loaded.sigma_global
    # One CPU-timed FNO pair per (sim, lead). FNO cost does not depend on t_s;
    # cycling it only spreads the timed pairs over the trajectory.
    timed_s = np.array([[sid % (n_snap - k) for k in leads] for sid in range(N)])

    truth = np.empty((N, n_snap, Nx, Ny), np.float32)
    cpu_pred = np.empty((N, L, Nx, Ny), np.float32)
    fv_setup = np.empty((R, N))
    fv_cum = np.empty((R, N, n_snap))
    fno_core = np.empty((R, N, L))
    fno_total = np.empty_like(fno_core)

    with threadpool_limits(limits=1), torch.inference_mode():
        warm = warm_params[0]
        for _ in range(3):
            solver = spec.configure_solver(warm, warm_setup['base_kwargs'])
            snaps, _ = fv_trajectory(solver, warm['T0'], stride)
        _, _, _, hist = solver.solve(T0=warm['T0'], store_trajectory=True)
        np.testing.assert_array_equal(snaps, hist[::stride])
        warm_ds = dataset(warm_params, np.stack([warm['T0'], warm['T0']])[None],
                          t_grid[[0, -1]], warm_setup, spec, loaded)
        warm_batch = {k: v.unsqueeze(0) for k, v in warm_ds[0].items()}
        for _ in range(5):
            model(warm_batch['spatial'], warm_batch['cond_static'], warm_batch['forcing_seq'])

        for rep in range(R):
            for sid, param in enumerate(params):
                start = time.perf_counter_ns()
                solver = spec.configure_solver(param, setup['base_kwargs'])
                snaps, stamps = fv_trajectory(solver, param['T0'], stride)
                fv_setup[rep, sid] = (stamps[0] - start) * 1e-9
                fv_cum[rep, sid] = (stamps - stamps[0]) * 1e-9
                if rep == 0:
                    assert len(solver.t) == n_steps + 1 and np.isclose(solver.t[-1], T_FINAL)
                    truth[sid] = snaps
                else:
                    np.testing.assert_array_equal(snaps.astype(np.float32), truth[sid])

                # Inference setup receives only the source state and the known forcing.
                for li, k in enumerate(leads):
                    s = int(timed_s[sid, li])
                    j = s + int(k)
                    start = time.perf_counter_ns()
                    ds = dataset([param], np.stack([truth[sid, s], truth[sid, s]])[None],
                                 t_grid[[s, j]], setup, spec, loaded)
                    batch = {key: v.unsqueeze(0) for key, v in ds[0].items()}
                    core_start = time.perf_counter_ns()
                    pred = model(batch['spatial'], batch['cond_static'], batch['forcing_seq'])
                    core_end = time.perf_counter_ns()
                    pred_K = pred * (sigma + T_EPS) + mu
                    end = time.perf_counter_ns()
                    fno_core[rep, sid, li] = (core_end - core_start) * 1e-9
                    fno_total[rep, sid, li] = (end - start) * 1e-9
                    if rep == 0:
                        cpu_pred[sid, li] = pred_K[0, ..., 0].numpy()
                    else:
                        np.testing.assert_array_equal(pred_K[0, ..., 0].numpy(), cpu_pred[sid, li])
                if rep == 0 and (sid + 1) % 20 == 0:
                    print(f'  round 1: {sid + 1}/{N} sims, {time.perf_counter() - started:.0f} s', flush=True)
            print(f'Timing round {rep + 1}/{R}: FV full horizon {(fv_setup[rep] + fv_cum[rep, :, -1]).mean():.4f} s, '
                  f'FNO {fno_total[rep].mean():.4f} s', flush=True)
        pools = threadpool_info()

    print(f'Error sweep over all pairs on {device}.', flush=True)
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count() or 1)
    eval_model = load_checkpoint(str(checkpoint), device=str(device)).model
    full = dataset(params, truth, t_grid, setup, spec, loaded)
    pairs = np.array(full._pairs, dtype=np.int64)
    timed = {(sid, int(timed_s[sid, li]), int(timed_s[sid, li] + k)): li
             for sid in range(N) for li, k in enumerate(leads)}
    loader = DataLoader(full, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=args.num_workers)
    scale = float(sigma) + T_EPS
    mse_K = np.empty(len(pairs))
    sse_norm = np.empty(len(pairs))
    sst_norm = np.empty(len(pairs))
    device_diff_K = 0.0
    device_rmse_rel_diff = 0.0
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            pred = eval_model(batch['spatial'].to(device), batch['cond_static'].to(device),
                              batch['forcing_seq'].to(device)).cpu().numpy().astype(np.float64)
            y = batch['Y'].numpy().astype(np.float64)
            rows = slice(cursor, cursor + len(y))
            diff = pred - y
            sse_norm[rows] = np.sum(diff ** 2, axis=(1, 2, 3))
            sst_norm[rows] = np.sum(y ** 2, axis=(1, 2, 3))
            mse_K[rows] = np.mean((diff * scale) ** 2, axis=(1, 2, 3))
            for n, key in enumerate(map(tuple, pairs[rows].tolist())):
                if key in timed:
                    sid, _, j = key
                    timed_pred = cpu_pred[sid, timed[key]].astype(np.float64)
                    pred_K = pred[n, ..., 0] * scale + mu
                    device_diff_K = max(device_diff_K, float(np.max(np.abs(pred_K - timed_pred))))
                    cpu_rmse = np.sqrt(np.mean((timed_pred - truth[sid, j]) ** 2))
                    device_rmse_rel_diff = max(device_rmse_rel_diff,
                                               abs(np.sqrt(mse_K[cursor + n]) - cpu_rmse) / cpu_rmse)
            cursor += len(y)
    assert cursor == len(pairs)

    rise = float(np.sqrt(sigma ** 2 + (mu - T_REF) ** 2))
    n_check = min(args.check_sims, N)
    sub = dataset(params[:n_check], truth[:n_check], t_grid, setup, spec, loaded)
    reference = evaluate(eval_model, DataLoader(sub, batch_size=args.batch_size, collate_fn=collate_fn),
                         device, x_grid=setup['grids']['x_grid'], use_per_sample_interface=True)
    check = pairs[:, 0] < n_check
    np.testing.assert_allclose(
        [100 * np.sqrt(mse_K[check].mean()) / rise, 100 * np.sqrt(sse_norm[check].sum() / sst_norm[check].sum())],
        [reference['gnrmse_pct'], reference['rel_l2_norm']], rtol=1e-4)
    device_agrees = device_rmse_rel_diff <= CPU_DEVICE_RMSE_RTOL

    lead_of_pair = pairs[:, 2] - pairs[:, 1]
    sim_gnrmse = np.empty((N, L))
    lead_rows = []
    # Mean cost of advancing k snapshots, averaged over every admissible source snapshot.
    fv_core_lead = np.stack([(fv_cum[..., k:] - fv_cum[..., :-k]).mean(-1) for k in leads], axis=-1)
    fv_total_lead = fv_core_lead + fv_setup[..., None]
    for li, k in enumerate(leads):
        sel = lead_of_pair == k
        counts = np.bincount(pairs[sel, 0], minlength=N)
        sim_gnrmse[:, li] = 100 * np.sqrt(np.bincount(pairs[sel, 0], weights=mse_K[sel], minlength=N) / counts) / rise
        p10, p25, p50, p75, p90 = np.percentile(sim_gnrmse[:, li], [10, 25, 50, 75, 90])
        times = {name: arr[..., li] for name, arr in (('fv_core', fv_core_lead), ('fv_total', fv_total_lead),
                                                     ('fno_core', fno_core), ('fno_total', fno_total))}
        row = dict(lead_s=float(t_grid[k]), fv_steps=int(k * stride), n_pairs=int(sel.sum()))
        for name, arr in times.items():
            row[f'{name}_s'] = float(arr.mean())
            row[f'{name}_std_s'] = float(arr.mean(0).std())
        row.update(
            speedup_core=row['fv_core_s'] / row['fno_core_s'],
            speedup_total=row['fv_total_s'] / row['fno_total_s'],
            pooled_gnrmse_pct=float(100 * np.sqrt(mse_K[sel].mean()) / rise),
            gnrmse_p10_pct=p10, gnrmse_p25_pct=p25, gnrmse_median_pct=p50,
            gnrmse_p75_pct=p75, gnrmse_p90_pct=p90,
            pooled_rmse_K=float(np.sqrt(mse_K[sel].mean())),
            pooled_rel_l2_norm_pct=float(100 * np.sqrt(sse_norm[sel].sum() / sst_norm[sel].sum())),
        )
        lead_rows.append(row)
    write_csv(out / 'per_lead.csv', lead_rows)

    pair_rows = []
    for (sid, s, j), mse, sse, sst in zip(pairs.tolist(), mse_K, sse_norm, sst_norm):
        p = params[sid]
        pair_rows.append(dict(
            sim_id=sid, ic_family=p['ic_family'], R_c=p['R_c'], interface_x=p['interface_x'],
            s=s, j=j, t_source_s=float(t_grid[s]), t_target_s=float(t_grid[j]),
            lead_s=float(t_grid[j - s]), rmse_K=float(np.sqrt(mse)),
            gnrmse_pct=float(100 * np.sqrt(mse) / rise), rel_l2_norm_pct=float(100 * np.sqrt(sse / sst)),
            trained_snapshots=bool(s in trained and j in trained)))
    write_csv(out / 'per_pair.csv', pair_rows)

    speedup_total = np.array([r['speedup_total'] for r in lead_rows])
    above = np.flatnonzero(speedup_total >= 1.0)
    result = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'checkpoint_epoch': ckpt.get('epoch'), 'checkpoint_training_seed': ckpt.get('seed'),
        'experiment': loaded.config.get('experiment', {}).get('name'), 'config': loaded.config,
        'training_population_hash': ckpt['normalization_provenance']['training_population_hash'],
        'training_data': training_data,
        'training_snapshot_indices': sorted(trained),
        'sampling': 'Original B2 Latin hypercube over (R_c, interface_x) with per-sim IC family, fresh RNG family.',
        'benchmark': 'interfaces', 'benchmark_label': 'B2: Varying interface + ITR',
        'representation': loaded.config['benchmark'].get('representation'),
        'num_simulations': N, 'generator_seed': args.seed, 'generator_seed_streams': resolve_seed_streams(args.seed),
        'ic_family_counts': {f: sum(p['ic_family'] == f for p in params) for f in sorted({p['ic_family'] for p in params})},
        'fv_dt_s': args.dt, 'fv_steps_total': n_steps, 'snapshot_stride': stride, 'snapshot_dt_s': SNAPSHOT_DT,
        't_final_s': T_FINAL, 'ramp_seconds': RAMP_SECONDS, 'grid': [Nx, Ny],
        'lead_times_s': [float(t_grid[k]) for k in leads], 'num_pairs': int(len(pairs)),
        'timing_repeats_per_simulation': R, 'warmup_fv_solves': 3, 'warmup_fno_forwards': 5,
        'warmup_seed': args.seed + 1, 'batch_size_timing': 1, 'torch_threads_timing': 1,
        'threadpools': pools, 'hardware': cpu_name(), 'platform': platform.platform(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'sweep_device': str(device),
        'sweep_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'mu_train_K': mu, 'sigma_train_K': sigma, 'training_rise_rms_K': rise,
        'mean_fno_forward_seconds': float(fno_core.mean()),
        'mean_fno_setup_and_predict_seconds': float(fno_total.mean()),
        'mean_fv_setup_seconds': float(fv_setup.mean()),
        'mean_fv_step_seconds': float(fv_cum[..., -1].mean() / n_steps),
        'fv_full_horizon_solve_seconds': float(fv_cum[..., -1].mean()),
        'speedup_total_at_max_lead': lead_rows[-1]['speedup_total'],
        'speedup_core_at_max_lead': lead_rows[-1]['speedup_core'],
        'first_lead_with_speedup_total_at_least_1_s': float(t_grid[leads[above[0]]]) if above.size else None,
        'pooled_gnrmse_pct_all_pairs': float(100 * np.sqrt(mse_K.mean()) / rise),
        'metric_scope': 'Direct prediction (no rollout) for every (t_s, t_j) pair on the 31 saved snapshots; FV is '
                        'the reference; fixed training normalization. gnRMSE = 100 * RMSE_K / training rise RMS. '
                        'Bands are across simulations, each pooled over its pairs at that lead.',
        'timing_scope': 'CPU single thread, batch one, timing interleaved FV/FNO per simulation. FV time for a lead '
                        'is the measured cost of the steps between two snapshots, averaged over all source '
                        'snapshots; core excludes solver assembly/factorization, total includes it. FNO core is one '
                        'forward pass; total adds input encoding and denormalization. Excludes loading, warmup, '
                        'sampling, disk I/O and metrics.',
        'validation': 'Weights match checkpoint; timestamped FV loop equals FVSolver2D.solve; repeated FV and FNO '
                      'results bitwise deterministic; NumPy metrics agree with evaluate() on the check sims; device '
                      'sweep checked against the timed CPU predictions (cpu_vs_device).',
        'evaluator_cross_check': {'num_sims': n_check, 'gnrmse_pct': reference['gnrmse_pct'],
                                  'rel_l2_norm': reference['rel_l2_norm']},
        'cpu_vs_device': {'max_abs_diff_K': device_diff_K, 'max_rmse_rel_diff': device_rmse_rel_diff,
                          'rmse_rtol': CPU_DEVICE_RMSE_RTOL, 'passed': bool(device_agrees)},
        'elapsed_seconds': time.perf_counter() - started,
    }
    (out / 'results.json').write_text(json.dumps(result, indent=2, default=jsonable))
    (out / 'simulation_parameters.json').write_text(
        json.dumps([{k: v for k, v in p.items() if k != 'T0'} for p in params], indent=2, default=jsonable))
    np.savez_compressed(out / 'timings.npz', t_grid=t_grid, leads=leads, timed_source_index=timed_s,
                        fv_setup_seconds=fv_setup, fv_cumulative_seconds=fv_cum,
                        fno_forward_seconds=fno_core, fno_total_seconds=fno_total,
                        sim_gnrmse_pct=sim_gnrmse)
    print(json.dumps({k: v for k, v in result.items() if k.startswith(('mean_', 'pooled_', 'speedup_', 'fv_full', 'first_'))},
                     indent=2), flush=True)
    # Raised only after writing, so the timing survives a device-precision failure.
    if not device_agrees:
        raise SystemExit(f'{device} sweep moves a timed pair RMSE by {100 * device_rmse_rel_diff:.2f}% relative to '
                         f'its CPU prediction (> {100 * CPU_DEVICE_RMSE_RTOL:g}%); error columns in {out} are suspect.')


def read_columns(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def plot(out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from visual.pub import style

    col = read_columns(out / 'per_lead.csv')
    meta = json.loads((out / 'results.json').read_text())
    dt, rise = meta['fv_dt_s'], meta['training_rise_rms_K']
    lead = col['lead_s']
    fno_color, fv_color, key_color = style.benchmark_color('interfaces'), '0.2', '0.45'
    marker = dict(marker='o', markersize=2.5)

    with style.pub_style():
        fig, axes = plt.subplots(1, 3, figsize=style.figsize('two_col', row_height='std'),
                                 constrained_layout=True)
        ax = axes[0]
        ax.plot(lead, col['fv_total_s'], color=fv_color, **marker)
        ax.plot(lead, col['fv_core_s'], color=fv_color, linestyle='--')
        ax.plot(lead, col['fno_total_s'], color=fno_color, **marker)
        ax.plot(lead, col['fno_core_s'], color=fno_color, linestyle='--')
        ax.set_yscale('log')
        ax.set_ylabel('CPU time per prediction [s]')
        top = ax.secondary_xaxis('top', functions=(lambda t: t / dt, lambda n: n * dt))
        top.set_xlabel(f'FV time steps ($\\Delta t$ = {dt:g} s)')

        ax = axes[1]
        ax.plot(lead, col['speedup_total'], color=fno_color, **marker)
        ax.plot(lead, col['speedup_core'], color=fno_color, linestyle='--')
        style.add_reference_line(ax, 1.0, label='Break even')
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel('Speedup, FV / FNO [-]')

        # Below the panels, so its placement does not depend on where the curves fall.
        handles = [
            Line2D([], [], color=fv_color, **marker, label='FV solver'),
            Line2D([], [], color=fno_color, **marker, label='FNO'),
            Line2D([], [], color=key_color, label='End to end'),
            Line2D([], [], color=key_color, linestyle='--',
                   label='Core (FV time stepping, FNO forward pass)'),
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)

        ax = axes[2]
        ax.fill_between(lead, col['gnrmse_p10_pct'], col['gnrmse_p90_pct'], color=fno_color,
                        alpha=0.2, linewidth=0, label='Simulation P10-P90')
        ax.plot(lead, col['pooled_gnrmse_pct'], color=fno_color, **marker, label='Pooled over pairs')
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel('FNO field error, gnRMSE [%]')
        right = ax.secondary_yaxis('right', functions=(lambda p: p * rise / 100.0, lambda K: K * 100.0 / rise))
        right.set_ylabel('RMSE [K]')
        ax.legend(loc='best')

        for ax in axes:
            ax.set_xlabel('Prediction lead time [s]')
            ax.set_xlim(0.0, lead[-1] + 0.005)
            ax.grid(True)
        style.panel_letters(axes)
        style.save(fig, out, FIGURE_KEY)


def main(argv=None):
    args = parse_args(argv)
    if not args.plot_only:
        measure(args)
    if not args.no_plot:
        plot(args.out)


if __name__ == '__main__':
    main()
