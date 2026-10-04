import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data.dataset import SnapshotPairDataset, split_sim_ids
from data.generate_dataset import build_base_setup, build_balanced_ic_families
from problems.registry import get_problem
from src.operators.cvit import ForcingICCViT
from src.physics.init_conditions import IC_FAMILIES
from src.operators.train import PICViTExponentialScheduler, build_scheduler
from src.operators.train_pino import (
    build_cvit, build_screen_model, model_inputs, build_forcing_image, cn_objective, cvit_gate,
    decode_grid, freeze_development_normalization, load_pino_inputs,
    make_boundary, matched_stream, require_gate, run_screen, anchored_previous,
    prefix_cn_objective, development_references,
    step_forcing_profile, evaluation_query_times, autoregressive_step,
)
from src.physics.boundary_forcing import reconstruct_qL
from src.physics.fv_preconditioner import ExactCNInverse
from src.physics.fv_residual import build_homogeneous_cn_geom


def tiny_model():
    return ForcingICCViT(emb_dim=16, num_heads=2, depth_enc=1, depth_dec=1,
                         ic_grid_size=(4, 4), ic_patch_size=2,
                         forcing_grid_size=(4, 8), forcing_patch_size=2,
                         t_right_tilde=1.7, t_final=0.3)


def config(model_type='cvit', residual_mode='one_step'):
    cfg = yaml.safe_load(Path('conf/config.yaml').read_text())
    cfg['paths'] = {}
    cfg['training'].update(device='cpu', optimizer='Adam', weight_decay=0.0, learning_rate=0.0005,
                           scheduler=dict(type='PICViTExponential', decay_every=500, decay_rate=0.95, min_lr=1e-5))
    cfg['physics_test']['cvit'].update(emb_dim=16, num_heads=2, depth_enc=1, depth_dec=1,
                    ic_patch_size=2, forcing_patch_size=2, forcing_grid_size=[4, 8])
    cfg['physics_test']['pino'].update(fixed_updates=2, fixed_validate_every=1,
                    query_chunk_size=5, evaluation_times=[0, 0.005, 0.01])
    cfg['physics_test']['model_type'] = model_type
    cfg['physics_test']['pino'].update(residual_mode='autoregressive' if residual_mode == 'mg' else residual_mode,prefix_steps=2)
    if residual_mode == 'space_time':
        cfg['physics_test']['pino'].update(controls=['raw','spatial_exact','exact'],checkpoint_keep_all=False)
    elif residual_mode in {'autoregressive', 'mg'}:
        cfg['physics_test']['pino'].update(controls=['mg' if residual_mode == 'mg' else 'exact'],checkpoint_keep_all=False,ic_weight=0.0)
        cfg['physics_test']['mg']['solver']['coarse_max_unknowns'] = 4
    if model_type == 'fno':
        cfg['model']['parameters'] = dict(modes1=2, modes2=2, width=8, n_layers=1,
                    cond_hidden=16, temporal_hidden=8, forcing_embed_dim=8,
                    forcing_spatial_dim=2, padding_mode='replicate',
                    cin_exclude_padding=True, hard_right_dirichlet=True)
        if residual_mode in {"autoregressive", "mg"}:
            cfg["model"]["parameters"].update(forcing_spatial_mode="boundary_extender",
                forcing_cond_mode="spatial_only", forcing_extender_heads=2, forcing_extender_grid_size=4)
    return cfg


def prescribed_inputs(n=4):
    setup = build_base_setup(20, 1, nx=n, ny=n, t_final=0.01)
    spec = get_problem('diffusion_forcing_single')
    setup['time_cfg']['ic_family_assignment'] = build_balanced_ic_families(20, list(IC_FAMILIES))
    params = spec.sample_sim_params(np.random.default_rng(0), np.random.default_rng(1),
                                   setup['grids'], setup['time_cfg'])
    return dict(spec=spec, params=np.array(params, dtype=object), x=setup['grids']['x_grid'],
                y=setup['grids']['y_grid'], dt=0.005, t_final=0.01, ramp_seconds=0.01,
                mu_global=295, sigma_global=10), setup


def save_inputs(tmp_path, inputs):
    for name, value in {'x_grid':inputs['x'], 'y_grid':inputs['y'], 't_grid':[0,0.005,0.01],
                        'dt':inputs['dt'], 'ramp_seconds':inputs['ramp_seconds'],
                        'sim_params':inputs['params'], 'meta':dict(problem_version=inputs['spec'].problem_version, ic_mode='varying')}.items():
        np.save(tmp_path / f'{name}.npy', value, allow_pickle=True)
    norm = tmp_path / 'normalization.json'
    norm.write_text(json.dumps(dict(mu_global=295, sigma_global=10, source='training_reference_trajectories')))
    return norm


def test_restored_model_hard_wall_chunking_and_both_encoder_gradients():
    torch.manual_seed(12)
    model = tiny_model()
    forcing, ic = torch.randn(2,1,4,8), torch.randn(2,1,4,4)
    latent = model.encode(forcing,ic)
    coords = torch.rand(1,16,2)
    coords[:, -1, 0] = 1
    full = decode_grid(model,latent,coords,[0.05,0.1],4,4,16)
    chunked = decode_grid(model,latent,coords,[0.05,0.1],4,4,3)
    torch.testing.assert_close(full,chunked)
    torch.testing.assert_close(full[:,-1,-1], torch.full((2,),1.7),rtol=0,atol=0)
    full.square().mean().backward()
    for branch in (model.forcing_encoder, model.ic_encoder):
        assert branch.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.decoder.time_film.net[-1].weight.grad.abs().sum() > 0


def test_cvit_accepts_only_exact_patch_tiling():
    with pytest.raises(ValueError,match='divisible'):
        ForcingICCViT(ic_grid_size=(11,11),ic_patch_size=10)


def test_picvit_scheduler_update_and_resume():
    parameter = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.Adam([parameter],lr=0.0005)
    scheduler = build_scheduler(config(),opt)
    assert scheduler.step_unit == 'update'
    for _ in range(500):
        scheduler.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.0005 * 0.95)
    saved = scheduler.state_dict()
    restored = PICViTExponentialScheduler(torch.optim.Adam([parameter],lr=0.0005),0.0005,500,0.95,1e-5)
    restored.load_state_dict(saved)
    assert restored.next_update == 500
    restored.step()
    scheduler.step()
    assert restored.state_dict() == scheduler.state_dict()
    saved['decay_rate'] = 0.9
    with pytest.raises(ValueError,match='resume mismatch'):
        restored.load_state_dict(saved)


def test_homogeneous_problem_dataset_and_solver_contract():
    inputs, setup = prescribed_inputs()
    spec = inputs['spec']
    solver = spec.configure_solver(inputs['params'][0],setup['base_kwargs'])
    assert len(solver.layers) == 1
    assert solver.layers[0].k == 1
    spec.validate_schema(inputs['params'],np.arange(20))
    trajectories = np.broadcast_to(np.stack([p['T0'] for p in inputs['params']])[:,None],(20,3,4,4)).copy()
    ds = SnapshotPairDataset(trajectories,np.array([0,0.005,0.01]),inputs['x'],inputs['y'],
             np.arange(20),inputs['params'],mu_global=295,sigma_global=10,n_snapshots=3,noise_std=0,problem=spec,
             dt=inputs['dt'],ramp_seconds=inputs['ramp_seconds'])
    item = ds[0]
    assert item['spatial'].shape == (4,4,4)
    assert item['cond_static'].shape == (1,)
    assert item['forcing_seq'].shape == (128,3)
    assert item['T_stats'].shape == (2,)
    stale = copy.deepcopy(inputs['params'])
    stale[0]['R_c'] = 0.2
    with pytest.raises(ValueError,match='forbids'):
        spec.validate_schema(stale,np.arange(20))


def test_input_only_loader_never_opens_future_references(tmp_path, monkeypatch):
    inputs,_ = prescribed_inputs()
    norm = save_inputs(tmp_path,inputs)
    original = np.load
    def guarded(path,*args,**kwargs):
        assert Path(path).name != 'trajectories.npy'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    loaded = load_pino_inputs(tmp_path,norm)
    assert loaded['sigma_global'] == 10
    (tmp_path/'meta.npy').unlink()
    with pytest.raises(ValueError,match='meta.npy'):
        load_pino_inputs(tmp_path,norm)


def test_forcing_image_matches_prescribed_flux_at_ramp_and_shutoff():
    inputs,_ = prescribed_inputs()
    p = inputs['params'][0]
    y,t = np.linspace(0,1,4),np.array([0,0.005,0.01,0.15,0.2,0.3])
    image = build_forcing_image([p],y,t,300,torch.device('cpu'),0.01)
    evaluator = reconstruct_qL(p['temporal_family'],p['temporal_params'],p['spatial_family'],p['spatial_params'],0.01)
    np.testing.assert_allclose(image[0,0].numpy(),evaluator.evaluate_grid(y,t)/300,rtol=1e-6,atol=1e-8)
    assert np.all(image[...,0].numpy() == 0)


def test_matched_stream_is_deterministic_and_first_interval_uses_true_ic():
    inputs,_ = prescribed_inputs()
    cfg = config()
    a,b = matched_stream(cfg,inputs,'fixed'),matched_stream(cfg,inputs,'fixed')
    np.testing.assert_array_equal(a['intervals'],b['intervals'])
    assert a['validation_ids'] == b['validation_ids']
    predicted = torch.ones(2,4,4,requires_grad=True)
    ic = torch.zeros_like(predicted)
    anchored = anchored_previous(predicted,ic,np.array([0,1]))
    assert torch.equal(anchored[0],ic[0])
    assert torch.equal(anchored[1],predicted[1])
    assert not anchored.requires_grad
    first_bc = make_boundary([a['problems'][0]],[0],inputs,torch.device('cpu'),torch.float64)
    geom = build_homogeneous_cn_geom(inputs['x'],inputs['y'],1,inputs['dt'],sigma_global=10)
    inverse = ExactCNInverse(geom)
    previous = torch.tensor(a['problems'][0]['T0'][None],dtype=torch.float64,requires_grad=True)
    following = previous.detach().clone().requires_grad_()
    loss,_,_ = cn_objective(previous,following,geom,first_bc,inverse,'exact')
    loss.backward()
    assert previous.grad is None
    assert following.grad is not None


def test_cvit_gate_requires_absolute_and_first_step_improvement_and_baselines():
    pino = config()['physics_test']['pino']
    exact = dict(trajectory_rmse_K=0.1,first_step_rmse_K=0.01,trajectory_sigma_error_percent=1,
                 copy_trajectory_rmse_K=2,equilibrium_trajectory_rmse_K=1,
                 copy_first_step_rmse_K=0.2,equilibrium_first_step_rmse_K=0.1)
    raw = dict(exact,trajectory_rmse_K=1,first_step_rmse_K=0.1)
    assert cvit_gate(raw,exact,pino,'fixed')['passed']
    assert not cvit_gate(raw,dict(exact,first_step_rmse_K=0.15),pino,'fixed')['passed']
    assert not cvit_gate(raw,dict(exact,copy_first_step_rmse_K=0.005),pino,'fixed')['passed']


def test_online_refuses_failed_fixed_gate_before_creating_artifacts(tmp_path):
    phase0 = tmp_path/'phase0'
    phase0.mkdir()
    (phase0/'final_metrics.json').write_text(json.dumps(dict(stage='phase0_direct_state',passed=True)))
    fixed = tmp_path/'fixed'
    fixed.mkdir()
    (fixed/'final_metrics.json').write_text(json.dumps(dict(stage='phase2_fixed_cvit',passed=False)))
    with pytest.raises(ValueError,match='did not pass'):
        run_screen(config(),tmp_path,tmp_path/'online',phase0,'online',fixed)
    assert not (tmp_path/'online').exists()


def test_mg_scientific_screen_requires_its_direct_state_gate_and_exact_baseline(tmp_path):
    phase4 = tmp_path/'phase4'
    phase4.mkdir()
    path = phase4/'final_metrics.json'
    path.write_text(json.dumps(dict(stage='phase0_direct_state',passed=True)))
    cfg = config(residual_mode='mg')
    with pytest.raises(ValueError, match='phase4_direct_state_mg'):
        run_screen(cfg,tmp_path,tmp_path/'mg',phase4)
    path.write_text(json.dumps(dict(stage='phase4_direct_state_mg',passed=True)))
    with pytest.raises(ValueError, match='completed exact prefix screen'):
        run_screen(cfg,tmp_path,tmp_path/'mg',phase4)
    assert not (tmp_path/'mg').exists()


def test_short_prefix_success_cannot_open_the_full_horizon_online_gate(tmp_path):
    phase0 = tmp_path/'phase0'
    phase0.mkdir()
    (phase0/'final_metrics.json').write_text(json.dumps(dict(stage='phase0_direct_state',passed=True)))
    prefix = tmp_path/'prefix'
    prefix.mkdir()
    (prefix/'final_metrics.json').write_text(json.dumps(dict(stage='phase2_fixed_cvit_prefix',passed=True)))
    with pytest.raises(ValueError,match='did not pass'):
        run_screen(config(residual_mode='space_time'),tmp_path,tmp_path/'online',phase0,'online',prefix)
    assert not (tmp_path/'online').exists()


def test_development_normalization_excludes_validation_and_test(tmp_path):
    ids = split_sim_ids(20,seed=0)
    labels = np.full((20,3,4,4),1e6,dtype=np.float32)
    labels[ids[0]] = np.arange(len(ids[0]) * 3 * 4 * 4).reshape(len(ids[0]),3,4,4)
    np.save(tmp_path/'trajectories.npy',labels)
    frozen = freeze_development_normalization(tmp_path,tmp_path/'normalization.json')
    assert frozen['mu_global'] == pytest.approx(labels[ids[0]].mean())
    assert frozen['sigma_global'] == pytest.approx(labels[ids[0]].std())
    assert frozen['source'] == 'training_reference_trajectories'


@pytest.mark.parametrize('model_type,residual_mode', [('cvit','one_step'),('fno','one_step'),
    ('cvit','space_time'),('fno','space_time'),('cvit','autoregressive'),('cvit','mg'),('fno','autoregressive'),('fno','mg')])
def test_two_update_end_to_end_smoke_cannot_open_a_later_gate(tmp_path, model_type, residual_mode):
    inputs,_ = prescribed_inputs()
    data = tmp_path/'data'
    data.mkdir()
    normalization = save_inputs(data,inputs)
    phase0 = tmp_path/'phase0'
    phase0.mkdir()
    (phase0/'final_metrics.json').write_text(json.dumps(dict(stage='phase0_direct_state',passed=True)))
    output = tmp_path/'smoke'
    result = run_screen(config(model_type,residual_mode),data,output,phase0,normalization_path=normalization,smoke=True)
    frozen=yaml.safe_load((output/'config_used.yaml').read_text())
    assert frozen['model']['type'] == ('ForcingICCViT' if model_type == 'cvit' else 'FNO2d')
    assert frozen['data']['num_sims'] == len(inputs['params'])
    assert frozen['paths']['data_dir'] == str(data.resolve())
    if model_type == 'cvit':
        assert frozen['physics_test']['protocol']['resolved_cvit_kwargs']['ic_grid_size'] == [4,4]
    else:
        assert frozen['physics_test']['protocol']['resolved_fno_dims']['in_channels'] == 4
        assert frozen['physics_test']['protocol']['input_encoding'] == (
            'current_state_and_exact_interval_average_flux_FNO_tokens' if residual_mode in {'autoregressive', 'mg'}
            else 'full_horizon_forcing_tokens_and_IC_query_time_cond')
    assert result['stage'] == 'phase1_smoke'
    assert not result['passed']
    for objective in config(model_type,residual_mode)['physics_test']['pino']['controls']:
        checkpoint = torch.load(output/objective/'checkpoint_000002.pt',weights_only=False)
        assert checkpoint['successful_updates'] == 2
        assert checkpoint['scheduler_state_dict']['next_update'] == 2
        assert checkpoint['mu_global'] == 295
    if residual_mode == 'space_time':
        assert (output/'spatial_exact'/'checkpoint_000002.pt').exists()
        assert not (output/'exact'/'checkpoint_000001.pt').exists()
        assert result['prefix_gate'] is not None
    elif residual_mode in {'autoregressive', 'mg'}:
        assert not (output/'raw').exists()
        assert not result['raw_comparison_performed']
        assert result['full_horizon_gate'] is None
        if model_type == 'cvit':
            assert not frozen['physics_test']['protocol']['resolved_cvit_kwargs']['use_time_query']
    with pytest.raises(ValueError,match='did not pass'):
        require_gate(output,'phase2_fixed_cvit')


@pytest.mark.parametrize('model_type,residual_mode', [('cvit','one_step'),('fno','one_step'),
    ('cvit','space_time'),('fno','space_time'),('cvit','autoregressive'),('cvit','mg'),('fno','autoregressive'),('fno','mg')])
def test_interrupted_screen_resume_replays_identical_updates(tmp_path,monkeypatch, model_type, residual_mode):
    import src.operators.train_pino as pino
    inputs,_ = prescribed_inputs()
    data=tmp_path/'data'
    data.mkdir()
    normalization=save_inputs(data,inputs)
    phase0=tmp_path/'phase0'
    phase0.mkdir()
    (phase0/'final_metrics.json').write_text(json.dumps(dict(stage='phase0_direct_state',passed=True)))
    baseline=tmp_path/'baseline'
    run_screen(config(model_type,residual_mode),data,baseline,phase0,normalization_path=normalization,smoke=True)
    interrupted=tmp_path/'interrupted'
    original=pino.evaluate_development
    calls=0
    def fail_after_second_update(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        return original(*args,**kwargs)
    monkeypatch.setattr(pino,'evaluate_development',fail_after_second_update)
    with pytest.raises(KeyboardInterrupt):
        run_screen(config(model_type,residual_mode),data,interrupted,phase0,normalization_path=normalization,smoke=True)
    monkeypatch.setattr(pino,'evaluate_development',original)
    pino.resume_screen(interrupted)
    for objective in config(model_type,residual_mode)['physics_test']['pino']['controls']:
        expected=torch.load(baseline/objective/'checkpoint_000002.pt',weights_only=False)
        actual=torch.load(interrupted/objective/'checkpoint_000002.pt',weights_only=False)
        assert actual['successful_updates'] == 2
        assert actual['scheduler_state_dict'] == expected['scheduler_state_dict']
        for name in actual['model_state_dict']:
            torch.testing.assert_close(actual['model_state_dict'][name],expected['model_state_dict'][name],rtol=0,atol=0)


def test_homogeneous_supervised_contract_can_run_the_shared_fno():
    from src.operators.fno2d import FNO2d
    dims=get_problem('diffusion_forcing_single').dims
    model=FNO2d(modes1=2,modes2=2,width=8,n_layers=1,in_channels=dims.in_channels,
                cond_static_dim=dims.cond_static_dim,temporal_token_dim=dims.temporal_token_dim,
                s_y_channel=dims.s_y_channel,use_forcing_time_aug=dims.use_forcing_time_aug,
                forcing_embed_dim=8,temporal_hidden=8,cond_hidden=16,forcing_spatial_dim=2)
    result=model(torch.randn(1,4,4,4),torch.tensor([[0.5]]),torch.randn(1,128,3))
    assert result.shape == (1,4,4,1)


def test_fno_screen_uses_full_horizon_forcing_original_ic_and_absolute_queries():
    cfg = config('fno')
    inputs, _ = prescribed_inputs()
    params = inputs['params'][:2].tolist()
    latent, ic = model_inputs(params, cfg, inputs, torch.device('cpu'))
    spatial, sequence, horizon = latent
    torch.testing.assert_close(spatial[..., 0], ic)
    trajectories = np.broadcast_to(np.stack([p['T0'] for p in params])[:, None],
                                   (2, 3, 4, 4)).copy()
    ds = SnapshotPairDataset(trajectories, np.array([0, .005, .01]), inputs['x'], inputs['y'],
             np.arange(2), np.array(params, dtype=object), mu_global=295, sigma_global=10,
             n_snapshots=3, noise_std=0, problem=inputs['spec'], dt=.005, ramp_seconds=.01)
    for sid in range(2):
        item = inputs['spec'].build_item(ds, sid, 0, 2)
        np.testing.assert_allclose(sequence[sid].numpy(), item['forcing_seq'], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(spatial[sid].numpy(), item['spatial'], rtol=1e-6, atol=1e-7)
    model = build_screen_model(cfg, inputs, torch.device('cpu'))
    captured = []
    handle = model.register_forward_pre_hook(lambda _, args: captured.append(args))
    output = decode_grid(model, latent, torch.empty(1,16,2), [.005,.01], 4, 4, 5)
    torch.testing.assert_close(captured[0][1], torch.tensor([[.5],[1.]]))
    assert captured[0][2] is sequence
    torch.testing.assert_close(output[:, -1], torch.full((2,4), .5), rtol=0, atol=0)
    output.square().mean().backward()
    assert model.linear_p.weight.grad.abs().sum() > 0
    assert model.temporal_encoder.lift.weight.grad.abs().sum() > 0
    handle.remove()


def test_prefix_exact_loss_has_same_cvit_parameter_gradients_as_trajectory_mse():
    cfg = config(residual_mode='space_time')
    inputs,_ = prescribed_inputs()
    params = inputs['params'][:1].tolist()
    forcing,ic = model_inputs(params,cfg,inputs,torch.device('cpu'))
    model = build_cvit(cfg,inputs,torch.device('cpu')).double()
    latent = model.encode(forcing.double(),ic[:,None].double())
    X,Y = np.meshgrid(inputs['x'],inputs['y'],indexing='ij')
    coords = torch.tensor(np.stack([X,Y],axis=-1).reshape(1,-1,2),dtype=torch.float64)
    predictions = [decode_grid(model,latent,coords,[t],4,4,5) for t in (.005,.01)]
    known_ic = ic.double()
    states = torch.stack([known_ic,*predictions],dim=1)
    geom = build_homogeneous_cn_geom(inputs['x'],inputs['y'],1,.005,sigma_global=10)
    inverse = ExactCNInverse(geom)
    boundaries = [make_boundary(params,[n],inputs,torch.device('cpu'),torch.float64) for n in range(2)]
    loss,_,_ = prefix_cn_objective(states,geom,boundaries,inverse,'exact')
    reference_params = copy.deepcopy(params)
    reference_params[0]['T0'] = known_ic[0].numpy()*10+295
    reference = development_references(inputs,reference_params,[0,.005,.01])
    reference = torch.tensor((reference[:,1:,:-1]-295)/10,dtype=torch.float64)
    direct = (states[:,1:,:-1]-reference).square().mean()
    torch.testing.assert_close(loss,direct,rtol=1e-11,atol=1e-11)
    actual = torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    expected = torch.autograd.grad(direct,tuple(model.parameters()))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)


def test_time_step_decoder_uses_current_state_and_spatial_queries_only():
    cfg=config(residual_mode='autoregressive')
    inputs,_=prescribed_inputs()
    params=inputs['params'][:1].tolist()
    model=build_cvit(cfg,inputs,torch.device('cpu'))
    forcing,ic=model_inputs(params,cfg,inputs,torch.device('cpu'))
    assert forcing is None
    forcing=step_forcing_profile(params,[0],cfg,inputs,torch.device('cpu'),ic.dtype)
    coords=torch.rand(1,16,2)
    coords[:,-4:,0]=1
    latent=model.encode(forcing,ic[:,None])
    assert latent.shape == (1, model.num_ic_tokens + 4, 16)
    full=decode_grid(model,latent,coords,None,4,4,16)
    chunked=decode_grid(model,latent,coords,None,4,4,3)
    torch.testing.assert_close(full,chunked)
    torch.testing.assert_close(full[:,-1],torch.full((1,4),.5),rtol=0,atol=0)
    other=model.encode(forcing,(ic+torch.randn_like(ic))[:,None])
    assert not torch.allclose(full,decode_grid(model,other,coords,None,4,4,16))
    assert not any('time_film' in n or 'fourier_t' in n for n,_ in model.named_parameters())
    full.square().mean().backward()
    assert model.forcing_encoder.net[0].weight.grad.abs().sum()>0
    assert model.ic_encoder.patch_embed.proj.weight.grad.abs().sum()>0


@pytest.mark.parametrize('interval',[0,1,3,39,40,50])
def test_step_forcing_profile_is_the_exact_local_cn_integral(interval):
    cfg=config(residual_mode='autoregressive')
    inputs,_=prescribed_inputs()
    params=inputs['params'][:2].tolist()
    profile=step_forcing_profile(params,[interval]*2,cfg,inputs,torch.device('cpu'),torch.float64)
    bc=make_boundary(params,[interval]*2,inputs,torch.device('cpu'),torch.float64)
    expected=bc.qL_int/(300*inputs['dt'])
    assert profile.shape == (2, len(inputs['y']))
    torch.testing.assert_close(profile,expected,rtol=1e-12,atol=1e-12)
    assert evaluation_query_times(cfg,inputs)==[0,.005,.01]


@pytest.mark.parametrize('num_tokens', [1, 4, 8])
def test_step_forcing_tokens_preserve_amplitude_sign_and_spatial_location(num_tokens):
    torch.manual_seed(42)
    cfg = config(residual_mode='autoregressive')
    cfg['physics_test']['cvit']['forcing_num_tokens'] = num_tokens
    inputs, _ = prescribed_inputs()
    model = build_cvit(cfg, inputs, torch.device('cpu')).double()
    profiles = torch.tensor([[0., 0., 0., 0.], [.5, .5, .5, .5],
                             [1., 1., 1., 1.], [-1., -1., -1., -1.],
                             [1., 0., 0., 0.], [0., 0., 0., 1.]],
                            dtype=torch.float64, requires_grad=True)
    state = torch.zeros(6, 1, 4, 4, dtype=torch.float64)
    coords = torch.tensor([[[0., 0.], [0., 1.], [.5, .5], [1., .5]]], dtype=torch.float64)
    latent = model.encode(profiles, state)
    assert latent.shape == (6, model.num_ic_tokens + num_tokens, 16)
    prediction = model.decode(latent, coords)
    for a, b in ((0, 1), (1, 2), (2, 3), (4, 5)):
        assert not torch.allclose(latent[a, :num_tokens], latent[b, :num_tokens])
        assert not torch.allclose(prediction[a, :-1], prediction[b, :-1])
    torch.testing.assert_close(prediction[:, -1], torch.full((6, 1), .5, dtype=torch.float64))
    prediction[:, :-1].square().sum().backward()
    assert torch.isfinite(profiles.grad).all()
    assert (profiles.grad.abs() > 1e-10).all()
    for parameter in model.forcing_encoder.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_step_forcing_profile_keeps_all_solver_nodes_and_signed_spatial_flux():
    cfg = config(residual_mode='autoregressive')
    inputs, _ = prescribed_inputs(n=10)
    params = copy.deepcopy(inputs['params'][:2].tolist())
    for p, center, amplitude in zip(params, [.2, .8], [200., -200.]):
        p.update(spatial_family='gaussian', spatial_params=dict(y_c=center, sigma_y=.1),
                 temporal_family='pulse_train',
                 temporal_params=dict(A_list=[amplitude], t_list=[0.], dt_list=[.1]))
    profile = step_forcing_profile(params, [3, 3], cfg, inputs, torch.device('cpu'), torch.float64)
    boundary = make_boundary(params, [3, 3], inputs, torch.device('cpu'), torch.float64)
    assert profile.shape == (2, 10)
    torch.testing.assert_close(profile, boundary.qL_int / (300 * inputs['dt']), rtol=1e-12, atol=1e-12)
    assert profile[0].max() > 0 and profile[1].min() < 0
    assert profile[0].argmax() == 2 and profile[1].argmin() == 7


def test_step_forcing_token_count_must_be_positive():
    with pytest.raises(ValueError, match='forcing_num_tokens'):
        ForcingICCViT(use_time_query=False, forcing_num_tokens=0)


def test_mlp_forcing_learns_cn_response_at_unseen_amplitudes():
    torch.manual_seed(42)
    cfg = config(residual_mode='autoregressive')
    inputs, setup = prescribed_inputs()
    model = build_cvit(cfg, inputs, torch.device('cpu'))
    xy = np.stack(np.meshgrid(inputs['x'], inputs['y'], indexing='ij'), axis=-1)
    coords = torch.tensor(xy.reshape(1, -1, 2), dtype=torch.float32)
    params = copy.deepcopy(inputs['params'][0])
    params.update(T0=np.full((4, 4), 300.), spatial_family='uniform', spatial_params={},
                  temporal_family='pulse_train',
                  temporal_params=dict(A_list=[300.], t_list=[0.], dt_list=[.1]))
    solver = inputs['spec'].configure_solver(params, setup['base_kwargs'])
    response = torch.tensor((solver.cn_step(params['T0'], .015) - 300) / 10,
                            dtype=torch.float32).reshape(1, -1, 1)
    amplitudes = torch.tensor([-1., -.5, 0., .5, 1.])
    profiles = amplitudes[:, None].expand(-1, 4)
    state = torch.full((5, 1, 4, 4), .5)
    targets = .5 + amplitudes[:, None, None] * response
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(200):
        loss = (model(profiles, state, coords) - targets).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        held_out = torch.tensor([-.75, -.25, .25, .75])
        prediction = model(held_out[:, None].expand(-1, 4), state[:4], coords)
        expected = .5 + held_out[:, None, None] * response
        # Scale by the forcing-induced temperature change, not the 300 K baseline.
        relative_error = (prediction - expected).norm() / (expected - .5).norm()
        response_error = ((prediction[-1] - prediction[0]) -
                          (expected[-1] - expected[0])).norm() / (expected[-1] - expected[0]).norm()
    assert relative_error < .08
    assert response_error < .08


@pytest.mark.parametrize('model_type', ['cvit', 'fno'])
def test_time_step_loss_and_parameter_gradients_match_cn_targets_for_each_current_state(model_type):
    cfg=config(model_type, residual_mode='autoregressive')
    inputs,setup=prescribed_inputs()
    params=inputs['params'][:1].tolist()
    model=build_screen_model(cfg,inputs,torch.device('cpu')).double()
    _,state=model_inputs(params,cfg,inputs,torch.device('cpu'))
    state=state.double()
    X,Y=np.meshgrid(inputs['x'],inputs['y'],indexing='ij')
    coords=torch.tensor(np.stack([X,Y],axis=-1).reshape(1,-1,2),dtype=torch.float64)
    geom=build_homogeneous_cn_geom(inputs['x'],inputs['y'],1,.005,sigma_global=10)
    inverse=ExactCNInverse(geom)
    solver=inputs['spec'].configure_solver(params[0],setup['base_kwargs'])
    losses,direct=[],[]
    for n in range(2):
        previous=state.detach()
        state=autoregressive_step(model,previous,params,[n],cfg,inputs,coords)
        bc=make_boundary(params,[n],inputs,torch.device('cpu'),torch.float64)
        loss,_,_=cn_objective(previous,state,geom,bc,inverse,'exact')
        target=solver.cn_step(previous[0].numpy()*10+295,n*.005)
        target=torch.tensor((target-295)/10,dtype=torch.float64)[None]
        losses.append(loss)
        direct.append((state[:,:-1]-target[:,:-1]).square().mean())
    loss=torch.stack(losses).mean()
    target_loss=torch.stack(direct).mean()
    torch.testing.assert_close(loss,target_loss,rtol=1e-11,atol=1e-11)
    actual=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    expected=torch.autograd.grad(target_loss,tuple(model.parameters()))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize('model_type', ['cvit', 'fno'])
def test_autoregressive_training_reencodes_detached_model_predictions(tmp_path,monkeypatch,model_type):
    import src.operators.train_pino as pino
    cfg=config(model_type, residual_mode='autoregressive')
    inputs,_=prescribed_inputs()
    data=tmp_path/'data'
    data.mkdir()
    normalization=save_inputs(data,inputs)
    phase0=tmp_path/'phase0'
    phase0.mkdir()
    (phase0/'final_metrics.json').write_text(json.dumps(dict(stage='phase0_direct_state',passed=True)))
    encoded,decoded=[],[]
    original_step=pino.autoregressive_step
    def capture_step(model,state,*args):
        if torch.is_grad_enabled():
            assert not state.requires_grad
            encoded.append(state.clone())
        result=original_step(model,state,*args)
        if torch.is_grad_enabled():
            decoded.append(result.detach().clone())
        return result
    monkeypatch.setattr(pino,'autoregressive_step',capture_step)
    output=tmp_path/'screen'
    run_screen(cfg,data,output,phase0,normalization_path=normalization,smoke=True)
    assert len(encoded)==len(decoded)==4
    torch.testing.assert_close(encoded[1],decoded[0],rtol=0,atol=0)
    torch.testing.assert_close(encoded[3],decoded[2],rtol=0,atol=0)
    torch.testing.assert_close(encoded[0],encoded[2],rtol=0,atol=0)
    assert not (output/'raw').exists()


def test_fno_step_uses_boundary_extension_and_preserves_hard_wall():
    torch.manual_seed(42)
    cfg = config('fno', residual_mode='mg')
    inputs, _ = prescribed_inputs()
    params = inputs['params'][:1].tolist()
    model = build_screen_model(cfg, inputs, torch.device('cpu'))
    _, state = model_inputs(params, cfg, inputs, torch.device('cpu'))
    x, y = np.meshgrid(inputs['x'], inputs['y'], indexing='ij')
    coords = torch.tensor(np.stack([x, y], axis=-1).reshape(1, -1, 2), dtype=state.dtype)
    predicted = autoregressive_step(model, state, params, [0], cfg, inputs, coords)
    torch.testing.assert_close(predicted[:, -1], torch.full_like(predicted[:, -1], .5), rtol=0, atol=0)
    assert not model._forcing_to_cond
    predicted[:, :-1].square().mean().backward()
    for branch in (model.temporal_encoder, model.boundary_extender, model.spectral_layers):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    cfg['model']['parameters']['hard_right_dirichlet'] = False
    with pytest.raises(ValueError, match='hard_right_dirichlet=true'):
        build_screen_model(cfg, inputs, torch.device('cpu'))


def test_exact_only_gate_checks_prefix_and_first_step_without_a_raw_comparison():
    pino=config(residual_mode='autoregressive')['physics_test']['pino']
    exact=dict(trajectory_rmse_K=.1,trajectory_sigma_error_percent=1,
               first_step_rmse_K=.01,first_step_sigma_error_percent=.1,
               copy_trajectory_rmse_K=2,equilibrium_trajectory_rmse_K=1,
               copy_first_step_rmse_K=2,equilibrium_first_step_rmse_K=1)
    result=cvit_gate(None,exact,pino,'fixed')
    assert result['passed'] and not result['raw_comparison_performed']
    assert result['improves_both'] is None and result['trajectory_raw_over_exact'] is None
    assert not cvit_gate(None,dict(exact,first_step_sigma_error_percent=6),pino,'fixed')['passed']
    assert not cvit_gate(None,dict(exact,trajectory_sigma_error_percent=6),pino,'fixed')['passed']
