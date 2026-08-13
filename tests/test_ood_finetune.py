"""Tests for scripts/run_ood_finetune.py (warm-start fine-tune on an OOD arm)."""

import copy

import pytest

from scripts.run_ood_finetune import (
    DIM_FIELDS,
    assert_dims_match,
    build_finetune_config,
    check_schedule_fits,
    inherited_provenance,
    parse_args,
    resolve_sim_split,
)


class _Dims:
    def __init__(self, **kwargs):
        for field in DIM_FIELDS:
            setattr(self, field, kwargs[field])


def _dims(**overrides):
    base = dict(in_channels=4, cond_static_dim=10, temporal_token_dim=2,
                s_y_channel=3, use_temporal_encoder=True, use_forcing_time_aug=True)
    base.update(overrides)
    return _Dims(**base)


def _model_params(**overrides):
    base = dict(in_channels=4, cond_static_dim=10, temporal_token_dim=2,
                s_y_channel=3, use_temporal_encoder=True, use_forcing_time_aug=True,
                modes1=16, modes2=16, width=64, forcing_spatial_mode="boundary_extender",
                forcing_extender_depth=2, forcing_cond_mode="spatial_only",
                temporal_samples=64)
    base.update(overrides)
    return base


def _source_conf():
    return {
        "experiment": {"name": "E11_source"},
        "benchmark": {"name": "forcing", "representation": "temporal_encoder",
                      "spatial_conditioning": "spatial_field_only"},
        "model": {"parameters": _model_params()},
        "training": {
            "epochs": 36, "learning_rate": 2.8e-3, "batch_size": 512,
            "weight_decay": 3e-5, "n_snapshots": 20, "n_snapshots_test": 40,
            "curriculum_warmup": 25, "validate_every": 5, "patience": 20,
            "checkpoint_metric": "val_rel_l2", "seeds": [32], "device": "auto",
            "num_workers": None, "init_from_checkpoint": None,
            "lead_cutoff_time": 0.2, "checkpoint_forgetting_ratio": 1.05,
            "scheduler": {"type": "RIGNOThreePhase", "warmup_fraction": 0.05,
                          "cosine_fraction": 0.85, "exp_fraction": 0.10,
                          "init_lr_ratio": 0.05, "cosine_floor_lr_ratio": 0.05,
                          "final_lr_ratio": 0.005},
            "run": {"run_dir": "/scratch/old"},
        },
        "data": {"trajectories.npy": "/scratch/old/trajectories.npy",
                 "x_grid_path": "/scratch/old/x_grid.npy",
                 "y_grid_path": "/scratch/old/y_grid.npy",
                 "t_grid_path": "/scratch/old/t_grid.npy",
                 "sim_params_path": "/scratch/old/sim_params.npy",
                 "num_sims": 8000},
    }


def _args(*extra):
    return parse_args(["/ckpt/fno2d_best.pt", "--data", "/arm", "--out", "/out", *extra])


FAMILIES = ["sin", "exp", "pulse_train", "exp_train"]


def _labels(num_sims=128, families=FAMILIES):
    """Round-robin labels, so a contiguous prefix would also look balanced.

    Balance therefore only distinguishes the stratified split from a shuffled
    one, which is the failure mode that matters: an unlucky random few-shot draw.
    """
    return [families[i % len(families)] for i in range(num_sims)]


def _composition(labels, ids):
    counts = {}
    for sim_id in ids:
        counts[labels[int(sim_id)]] = counts.get(labels[int(sim_id)], 0) + 1
    return counts


class TestSimSplit:
    def test_train_and_val_are_disjoint(self):
        labels = _labels()
        train, val, _ = resolve_sim_split(labels, 16, 16, seed=0)
        assert len(train) == 16 and len(val) == 16
        assert not set(train.tolist()) & set(val.tolist())

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_train_subset_is_balanced_across_temporal_families(self, n):
        labels = _labels()
        train, _, _ = resolve_sim_split(labels, n, 16, seed=0)
        counts = _composition(labels, train)
        assert set(counts) == set(FAMILIES)
        assert set(counts.values()) == {n // len(FAMILIES)}

    def test_val_set_is_balanced_too(self):
        labels = _labels()
        _, val, _ = resolve_sim_split(labels, 16, 16, seed=0)
        assert set(_composition(labels, val).values()) == {4}

    def test_train_subsets_are_nested_as_n_grows(self):
        # The whole point of the N sweep: moving 8 -> 16 adds target data rather
        # than redrawing a different cohort.
        labels = _labels()
        subsets = [
            set(resolve_sim_split(labels, n, 16, seed=0)[0].tolist())
            for n in (8, 16, 32, 64)
        ]
        for smaller, larger in zip(subsets, subsets[1:]):
            assert smaller < larger

    def test_val_ids_are_invariant_to_train_sims(self):
        labels = _labels()
        val_8 = resolve_sim_split(labels, 8, 16, seed=0)[1]
        val_64 = resolve_sim_split(labels, 64, 16, seed=0)[1]
        assert val_8.tolist() == val_64.tolist()

    def test_val_never_enters_any_train_subset(self):
        labels = _labels()
        val = set(resolve_sim_split(labels, 8, 16, seed=0)[1].tolist())
        for n in (8, 16, 32, 64):
            train = resolve_sim_split(labels, n, 16, seed=0)[0]
            assert not val & set(train.tolist())

    def test_split_is_deterministic_for_a_seed(self):
        labels = _labels()
        first = resolve_sim_split(labels, 16, 16, seed=3)
        second = resolve_sim_split(labels, 16, 16, seed=3)
        for a, b in zip(first, second):
            assert a.tolist() == b.tolist()

    def test_split_seed_draws_a_different_cohort(self):
        labels = _labels()
        a = resolve_sim_split(labels, 16, 16, seed=0)[0]
        b = resolve_sim_split(labels, 16, 16, seed=1)[0]
        assert a.tolist() != b.tolist()

    def test_uneven_label_counts_stay_as_balanced_as_possible(self):
        # 5 sims of one family, 20 of the others: the small family is exhausted
        # rather than silently dropping the request.
        labels = ["rare"] * 5 + ["common"] * 20
        train, _, _ = resolve_sim_split(labels, 16, 4, seed=0)
        counts = _composition(labels, train)
        assert counts["rare"] <= 5
        assert counts["rare"] + counts["common"] == 16

    def test_leftover_sims_become_the_discarded_test_split(self):
        labels = _labels(100)
        train, val, test = resolve_sim_split(labels, 48, 16, seed=0)
        assert len(test) == 36
        assert not set(test.tolist()) & (set(train.tolist()) | set(val.tolist()))

    def test_exactly_consumed_arm_falls_back_to_val_ids(self):
        labels = _labels(64)
        _, val, test = resolve_sim_split(labels, 48, 16, seed=0)
        assert test.tolist() == val.tolist()

    def test_oversubscribed_split_raises(self):
        with pytest.raises(ValueError, match="64 sims"):
            resolve_sim_split(_labels(64), 48, 32)

    @pytest.mark.parametrize("train_sims,val_sims", [(0, 4), (4, 0)])
    def test_empty_split_raises(self, train_sims, val_sims):
        with pytest.raises(ValueError, match=">= 1"):
            resolve_sim_split(_labels(64), train_sims, val_sims)


class TestDimsGuard:
    def test_matching_dims_pass(self):
        assert_dims_match(_dims(), _model_params())

    def test_drifted_dim_raises_naming_the_field(self):
        with pytest.raises(ValueError, match="cond_static_dim: checkpoint=11 spec=10"):
            assert_dims_match(_dims(), _model_params(cond_static_dim=11))

    def test_absent_field_is_not_compared(self):
        params = _model_params()
        del params["s_y_channel"]
        assert_dims_match(_dims(), params)


class TestFinetuneConfig:
    def test_architecture_and_benchmark_are_inherited_verbatim(self):
        source = _source_conf()
        config = build_finetune_config(
            source, checkpoint_path="/ckpt/fno2d_best.pt", data_dir="/arm",
            run_root="/out", args=_args())
        assert config["model"] == source["model"]
        assert config["benchmark"] == source["benchmark"]

    def test_source_conf_is_not_mutated(self):
        source = _source_conf()
        before = copy.deepcopy(source)
        build_finetune_config(source, checkpoint_path="/ckpt/fno2d_best.pt",
                              data_dir="/arm", run_root="/out", args=_args())
        assert source == before

    def test_optimization_is_overridden_for_a_warm_start(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/ckpt/fno2d_best.pt", data_dir="/arm",
            run_root="/out",
            args=_args("--epochs", "20", "--lr", "2e-4", "--batch-size", "32",
                       "--seed", "7", "--device", "mps"))
        training = config["training"]
        assert training["init_from_checkpoint"] == "/ckpt/fno2d_best.pt"
        assert training["epochs"] == 20
        assert training["learning_rate"] == pytest.approx(2e-4)
        assert training["batch_size"] == 32
        assert training["seeds"] == [7]
        assert training["device"] == "mps"
        assert training["run"]["run_dir"] == "/out"

    def test_curriculum_and_long_lead_machinery_are_disabled(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/ckpt/fno2d_best.pt", data_dir="/arm",
            run_root="/out", args=_args())
        training = config["training"]
        assert training["curriculum_warmup"] == 0
        assert training["lead_cutoff_time"] is None
        assert training["checkpoint_forgetting_ratio"] is None
        assert training["checkpoint_metric"] == "val_rel_l2"

    def test_lead_cutoff_opts_into_the_step0_reference_pass(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/ckpt/fno2d_best.pt", data_dir="/arm",
            run_root="/out", args=_args("--lead-cutoff-time", "0.15"))
        assert config["training"]["lead_cutoff_time"] == pytest.approx(0.15)
        # Selection must stay on plain val rel_l2, or the cutoff would silently
        # change which epoch is kept.
        assert config["training"]["checkpoint_metric"] == "val_rel_l2"
        assert config["training"]["checkpoint_forgetting_ratio"] is None

    def test_weight_decay_is_inherited_unless_overridden(self):
        inherited = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args())
        assert inherited["training"]["weight_decay"] == pytest.approx(3e-5)
        overridden = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--weight-decay", "1e-4"))
        assert overridden["training"]["weight_decay"] == pytest.approx(1e-4)

    def test_eval_snapshot_count_stays_inherited(self):
        # n_snapshots_test only reaches the discarded test loader here; rewriting
        # it would mislead a later run_eval.py on this run dir.
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--n-snapshots", "31"))
        assert config["training"]["n_snapshots"] == 31
        assert config["training"]["n_snapshots_test"] == 40

    def test_data_paths_point_at_the_finetune_arm(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm/data_sinusoid",
            run_root="/out", args=_args())
        assert config["data"]["trajectories.npy"] == "/arm/data_sinusoid/trajectories.npy"
        assert config["data"]["sim_params_path"] == "/arm/data_sinusoid/sim_params.npy"
        assert config["data"]["t_grid_path"] == "/arm/data_sinusoid/t_grid.npy"

    def test_experiment_name_derives_from_the_source_run(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args())
        assert config["experiment"]["name"] == "E11_source__ft_sinusoid"


class TestScratchBaseline:
    def test_scratch_drops_the_pretrained_weights_only(self):
        source = _source_conf()
        config = build_finetune_config(
            source, checkpoint_path="/ckpt/fno2d_best.pt", data_dir="/arm",
            run_root="/out", args=_args("--scratch", "--lr", "1.75e-4"))
        assert config["training"]["init_from_checkpoint"] is None
        # Everything that makes the comparison fair must survive.
        assert config["model"] == source["model"]
        assert config["benchmark"] == source["benchmark"]
        assert config["training"]["epochs"] == 20
        assert config["training"]["batch_size"] == 32

    def test_warm_start_still_loads_the_checkpoint(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/ckpt/fno2d_best.pt", data_dir="/arm",
            run_root="/out", args=_args())
        assert config["training"]["init_from_checkpoint"] == "/ckpt/fno2d_best.pt"

    def test_scratch_run_is_named_apart_from_the_finetune(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--scratch", "--lr", "1.75e-4"))
        assert config["experiment"]["name"] == "E11_source__scratch_sinusoid"

    def test_scratch_without_an_explicit_lr_is_rejected(self):
        # Silently reusing the warm-start default would build a strawman.
        with pytest.raises(SystemExit):
            _args("--scratch")

    def test_scratch_uses_the_same_split_as_the_finetune(self):
        labels = _labels()
        warm = _args("--train-sims", "16")
        cold = _args("--scratch", "--lr", "1.75e-4", "--train-sims", "16")
        a = resolve_sim_split(labels, warm.train_sims, warm.val_sims,
                              seed=warm.split_seed)
        b = resolve_sim_split(labels, cold.train_sims, cold.val_sims,
                              seed=cold.split_seed)
        assert a[0].tolist() == b[0].tolist()
        assert a[1].tolist() == b[1].tolist()


class TestScheduleGuard:
    def test_short_finetune_is_rejected_before_training(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--epochs", "2"))
        with pytest.raises(SystemExit, match="too short"):
            check_schedule_fits(config)

    def test_normal_finetune_length_passes(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--epochs", "20"))
        check_schedule_fits(config)

    def test_resolved_lr_curve_is_returned_for_inspection(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--epochs", "20", "--lr", "2e-5"))
        schedule = check_schedule_fits(config)
        assert schedule["peak_lr"] == pytest.approx(2e-5)
        total = (schedule["warmup_epochs"] + schedule["cosine_epochs"]
                 + schedule["exp_epochs"])
        assert total == 20
        assert 0.0 < schedule["transition_fraction_of_run"] < 1.0

    def test_non_rigno_scheduler_has_no_curve_to_report(self):
        source = _source_conf()
        source["training"]["scheduler"] = {"type": "CosineAnnealingLR"}
        config = build_finetune_config(
            source, checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--epochs", "20"))
        assert check_schedule_fits(config) is None


class TestConstantLR:
    def test_constant_lr_replaces_the_inherited_schedule(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--constant-lr", "--lr", "3e-7"))
        assert config["training"]["scheduler"] == {
            "type": "StepLR", "step_size": 1, "gamma": 1.0}
        assert config["training"]["learning_rate"] == pytest.approx(3e-7)

    def test_constant_lr_is_reported_as_a_flat_curve(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args("--constant-lr", "--lr", "3e-7"))
        schedule = check_schedule_fits(config)
        assert schedule["type"] == "constant"
        assert schedule["constant_lr"] == pytest.approx(3e-7)
        # A flat run has no phases; the banner must not go looking for them.
        assert "warmup_epochs" not in schedule

    def test_constant_lr_bypasses_the_phase_length_guard(self):
        """A 2-epoch RIGNO run is rejected; a 2-epoch flat run is legitimate."""
        args = _args("--constant-lr", "--lr", "3e-7", "--epochs", "2")
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=args)
        assert check_schedule_fits(config)["type"] == "constant"

    def test_inherited_schedule_is_the_default(self):
        config = build_finetune_config(
            _source_conf(), checkpoint_path="/c", data_dir="/arm", run_root="/out",
            args=_args())
        assert config["training"]["scheduler"]["type"] == "RIGNOThreePhase"


class TestNormalizationProvenance:
    def test_source_provenance_is_carried_forward_and_tagged(self):
        source = {"training_mean_K": 303.1, "training_population_std_K": 7.18,
                  "training_population_hash": "abc"}
        out = inherited_provenance({"normalization_provenance": source}, "/ckpt.pt")
        assert out["training_population_hash"] == "abc"
        assert out["training_mean_K"] == pytest.approx(303.1)
        assert out["inherited_from_checkpoint"] == "/ckpt.pt"
        assert source == {"training_mean_K": 303.1,
                          "training_population_std_K": 7.18,
                          "training_population_hash": "abc"}

    def test_absent_provenance_stays_absent(self):
        assert inherited_provenance({}, "/ckpt.pt") is None


class TestArgDefaults:
    def test_defaults_are_a_local_warm_start(self):
        args = _args()
        # Bracketed by pilots on this checkpoint, not scaled from the
        # from-scratch peak: 1e-8 was a no-op, 3e-6 was destructive.
        assert args.lr == pytest.approx(3e-7)
        assert not args.constant_lr
        assert args.epochs == 20
        assert args.seed == 42
        # The sweep starts at 16 sims, not 48: the interesting part of a
        # few-shot curve is the low-N end.
        assert args.train_sims == 16 and args.val_sims == 16
        assert args.split_seed == 0
        assert args.device in ("mps", "auto", "cuda")

    def test_split_sizes_reach_resolve_sim_split(self):
        args = _args("--train-sims", "12", "--val-sims", "4")
        train, val, _ = resolve_sim_split(
            _labels(16), args.train_sims, args.val_sims, seed=args.split_seed)
        assert len(train) == 12 and len(val) == 4
        assert not set(train.tolist()) & set(val.tolist())
