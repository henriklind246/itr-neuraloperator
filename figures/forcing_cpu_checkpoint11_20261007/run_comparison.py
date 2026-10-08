"""Paired CPU timing and final-field accuracy for ten fresh forcing cases."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

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
from data.dataset import SnapshotPairDataset, problem_from_config, T_EPS
from scripts.invert import load_checkpoint
from src.operators.eval import evaluate

OUT = Path(__file__).resolve().parent
CHECKPOINT = Path('/Users/henriklind/Downloads/fno2d_best (11).pt')
SEED = 20261007
N = 10
REPEATS = 7


def draw(spec, count, seed):
    seeds = resolve_seed_streams(seed)
    setup = build_base_setup(count, 2, lhs_seed=seeds['lhs'], forcing_profile_seed=seeds['forcing_profile'])
    params = spec.sample_sim_params(np.random.default_rng(seeds['ic_params']),
                                   np.random.default_rng(seeds['forcing_profile']),
                                   setup['grids'], setup['time_cfg'])
    return params, setup


def dataset(params, fields, spec, loaded):
    return SnapshotPairDataset(
        trajectories=fields, t_grid=np.array([0., .3], np.float32),
        x_grid=np.linspace(0, 1, 100).astype(np.float32),
        y_grid=np.linspace(0, 1, 100).astype(np.float32),
        sim_ids=np.arange(len(params)), sim_params=np.array(params, dtype=object),
        mu_global=loaded.mu_global, sigma_global=loaded.sigma_global,
        dt=.005, t_final=.3, ramp_seconds=.01, problem=spec,
    )


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    loaded = load_checkpoint(str(CHECKPOINT), device='cpu')
    model = loaded.model
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert all(torch.equal(v, ckpt['model_state'][k]) for k, v in model.state_dict().items())
    spec = problem_from_config(loaded.config)
    assert spec.name == 'forcing'
    pop = ckpt['normalization_provenance']['training_population']
    assert len(pop['x_grid']) == len(pop['y_grid']) == 100
    assert np.isclose(pop['t_grid'][-1], .3)
    params, setup = draw(spec, N, SEED)
    warm_params, warm_setup = draw(spec, 1, SEED + 1)
    fields = np.empty((N, 2, 100, 100), np.float32)
    fv_core = np.empty((REPEATS, N))
    fv_total = np.empty_like(fv_core)
    fno_core = np.empty_like(fv_core)
    fno_total = np.empty_like(fv_core)
    preds_norm = np.empty((N, 100, 100, 1), np.float32)
    with threadpool_limits(limits=1), torch.inference_mode():
        for _ in range(3):
            warm_solver = spec.configure_solver(warm_params[0], warm_setup['base_kwargs'])
            warm_solver.solve(T0=warm_params[0]['T0'], store_trajectory=False)
        warm_fields = np.stack([warm_params[0]['T0'], warm_params[0]['T0']])[None]
        warm_ds = dataset(warm_params, warm_fields, spec, loaded)
        warm_batch = {k: v.unsqueeze(0) for k, v in warm_ds[0].items()}
        for _ in range(5):
            model(warm_batch['spatial'], warm_batch['cond_static'], warm_batch['forcing_seq'])

        for rep in range(REPEATS):
            for sid, param in enumerate(params):
                start = time.perf_counter_ns()
                solver = spec.configure_solver(param, setup['base_kwargs'])
                core_start = time.perf_counter_ns()
                t, x, y, final = solver.solve(T0=param['T0'], store_trajectory=False)
                end = time.perf_counter_ns()
                fv_core[rep, sid] = (end - core_start) * 1e-9
                fv_total[rep, sid] = (end - start) * 1e-9
                assert len(t) == 61 and np.isclose(t[-1], .3)
                if rep == 0:
                    fields[sid] = np.stack([param['T0'], final.astype(np.float32)])
                else:
                    np.testing.assert_array_equal(final.astype(np.float32), fields[sid, 1])

                # Inference setup receives only the initial condition and known forcing.
                start = time.perf_counter_ns()
                inference_fields = np.stack([param['T0'], param['T0']])[None]
                ds = dataset([param], inference_fields, spec, loaded)
                batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}
                core_start = time.perf_counter_ns()
                pred = model(batch['spatial'], batch['cond_static'], batch['forcing_seq'])
                core_end = time.perf_counter_ns()
                pred_K = pred * (loaded.sigma_global + T_EPS) + loaded.mu_global
                end = time.perf_counter_ns()
                fno_core[rep, sid] = (core_end - core_start) * 1e-9
                fno_total[rep, sid] = (end - start) * 1e-9
                if rep == 0:
                    preds_norm[sid] = pred[0].numpy()
                else:
                    np.testing.assert_array_equal(pred[0].numpy(), preds_norm[sid])
            print(f'Timing round {rep + 1}/{REPEATS}: FV {fv_total[rep].mean():.6f}s, FNO {fno_total[rep].mean():.6f}s', flush=True)

        ds = dataset(params, fields, spec, loaded)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        batches = list(loader)
        reference_metrics = evaluate(model, loader, torch.device('cpu'), x_grid=x,
            prediction_batches=[(b, torch.from_numpy(preds_norm[i:i+1])) for i, b in enumerate(batches)])
        pools = threadpool_info()

    mu, sigma = loaded.mu_global, loaded.sigma_global
    rise = float(np.sqrt(sigma**2 + (mu - 300.)**2))
    prediction = preds_norm.astype(np.float64)[..., 0] * (sigma + T_EPS) + mu
    truth = fields[:, 1].astype(np.float64)
    error = prediction - truth
    rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    left = int(np.searchsorted(x, .5)) - 1
    jump_error = error[:, left, :] - error[:, left+1, :]
    jump_rmse = np.sqrt(np.mean(jump_error**2, axis=1))
    field_each = 100 * rmse / rise
    field_pooled = float(100 * np.sqrt(np.mean(error**2)) / rise)
    jump_each = 100 * jump_rmse / sigma
    np.testing.assert_allclose(field_pooled, reference_metrics['gnrmse_pct'], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(jump_each.mean(), reference_metrics['node_jump_gnrmse_pct'], rtol=3e-5, atol=3e-5)
    assert np.isfinite(prediction).all()
    rows=[]
    for sid, p in enumerate(params):
        rows.append(dict(sim_id=sid, temporal_family=p['temporal_family'], spatial_family=p['spatial_family'], R_c=p['R_c'],
            fv_solve_seconds=float(fv_core[:,sid].mean()), fv_setup_and_solve_seconds=float(fv_total[:,sid].mean()),
            fno_forward_seconds=float(fno_core[:,sid].mean()), fno_setup_and_predict_seconds=float(fno_total[:,sid].mean()),
            field_rmse_K=float(rmse[sid]), field_nrmse_pct=float(field_each[sid]),
            node_jump_rmse_K=float(jump_rmse[sid]), node_jump_nrmse_pct=float(jump_each[sid])))
    with (OUT/'per_simulation.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'checkpoint_epoch': ckpt['epoch'], 'checkpoint_training_seed': ckpt['seed'], 'config': loaded.config,
        'training_population_hash': ckpt['normalization_provenance']['training_population_hash'],
        'training_dataset_seed_evidence': 'Checkpoint omits generator seed; fresh seed is independent of repository default seed 0.',
        'benchmark':'forcing', 'representation':'temporal_encoder', 'num_simulations':N,
        'generator_seed':SEED, 'generator_seed_streams':resolve_seed_streams(SEED),
        'source_time_s':0., 'target_time_s':.3, 'fv_dt_s':.005, 'fv_steps':60, 'grid':[100,100],
        'timing_repeats_per_simulation':REPEATS, 'warmup_fv_solves':3, 'warmup_fno_forwards':5,
        'warmup_seed':SEED+1, 'batch_size':1, 'torch_threads':torch.get_num_threads(),
        'torch_interop_threads':torch.get_num_interop_threads(), 'threadpools':pools,
        'hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        'platform':platform.platform(), 'torch':torch.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__,
        'mu_train_K':mu,'sigma_train_K':sigma,'training_rise_rms_K':rise,
        'mean_fv_solve_seconds':float(fv_core.mean()), 'mean_fv_setup_and_solve_seconds':float(fv_total.mean()),
        'mean_fno_forward_seconds':float(fno_core.mean()), 'mean_fno_setup_and_predict_seconds':float(fno_total.mean()),
        'speedup_solve_over_forward':float(fv_core.mean()/fno_core.mean()),
        'speedup_with_setup':float(fv_total.mean()/fno_total.mean()),
        'pooled_field_nrmse_pct':field_pooled, 'mean_per_sim_field_nrmse_pct':float(field_each.mean()),
        'mean_node_jump_nrmse_pct':float(jump_each.mean()),
        'metric_scope':'Final field at t=0.3 s only; FV is the reference; fixed training normalization.',
        'timing_scope':'CPU single thread, batch one. Excludes loading, training, warmup, sampling, disk I/O, metrics. Core FV excludes assembly/factorization; core FNO excludes input encoding and denormalization. With-setup includes these costs. No trajectory storage.',
        'validation':'Weights match checkpoint exactly; all repeated results deterministic; NumPy metrics agree with existing evaluate().',
        'evaluator_metrics':{k:reference_metrics[k] for k in ['gnrmse_pct','node_jump_gnrmse_pct']},
    }
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'simulation_parameters.json').write_text(json.dumps([{k:v for k,v in p.items() if k!='T0'} for p in params],indent=2))
    np.savez_compressed(OUT/'fields_and_timings.npz', initial=fields[:,0], truth=truth, prediction=prediction,
        x_grid=x,y_grid=y,prediction_normalized=preds_norm, fv_solve_seconds=fv_core, fv_total_seconds=fv_total,
        fno_forward_seconds=fno_core,fno_total_seconds=fno_total)
    print(json.dumps({k:v for k,v in result.items() if k.startswith(('mean_', 'pooled_', 'speedup_'))},indent=2))


if __name__=='__main__':
    main()
