"""Fine-tune a trained forcing checkpoint on a few unseen-family (sinusoid) sims.

`run_ood_spatial_family.py` measures what a checkpoint does on the sinusoid
spatial family zero-shot. This script asks the next question: how much of that
gap closes when the model is allowed to see a handful of sinusoid trajectories?

The default full fine-tune updates every weight; the matched --scratch control
isolates whether pretraining transferred useful structure. The one selective
alternative, --trainable-scope boundary_extender, freezes the backbone and asks
whether adapting the existing boundary-to-domain forcing map is sufficient.

Only training lives here. Data generation and scoring are the existing study
script, unchanged:

  1. Generate a fine-tune arm DISJOINT from the arms the checkpoint was scored
     on, by reusing the study's own generator at a different --rng-seed. Size
     the pool once at 128 simulations. Their roles are locked permanently:
     16 base training, 16 validation, 48 optional training expansion, and 48
     final untouched test simulations:

       python scripts/run_ood_spatial_family.py <ckpt> \
           --out-dir ood_studies/sin_ft_data --rng-seed 23 --num-sims 128 \
           --no-id-baseline --no-uniform-anchor --skip-eval

  2. Fine-tune (this script). The shot count is in SIMULATIONS; the all-to-all
     snapshot expansion (190 pairs/sim at --n-snapshots 20) is not a count of
     independent target examples and must not be reported as one:

       python scripts/run_ood_finetune.py <ckpt> \
           --data ood_studies/sin_ft_data/data_sinusoid \
           --out ood_studies/sin_ft_run --train-sims 16 --val-sims 16

     D16, D32, and D64 are the only data conditions. D32 is exactly D16 plus
     the first 16 simulations of the locked expansion order; D64 adds the full
     48-simulation expansion cohort. Validation and final-test IDs never change.

  2a. Calibrate the LR at N=16 before the confirmation run, with --constant-lr, all trials
     starting from the same untouched checkpoint. Two pilots bracket it:
     1e-8 was a clean no-op (val 5.2992 -> 5.2940) and 3e-6 was destructive
     (6.18 -> 6.45 -> 9.18 -> 9.41), so the usable range is between them:

       for LR in 1e-7 3e-7 1e-6; do
         python scripts/run_ood_finetune.py <ckpt> --constant-lr --lr $LR \
             --data ood_studies/sin_ft_data/data_sinusoid \
             --out ood_studies/sin_lrcal_$LR --train-sims 16 --val-sims 16
       done

     --constant-lr matters because the destructive pilot kept degrading WHILE
     its LR decayed 3e-6 -> 1.13e-6 -> 1.5e-8. A decaying schedule confounds
     "this update scale is safe" with "the schedule got out of the way in
     time", and only the first question is being asked here. Put a decay back
     on afterwards, around a scale that is already known to be safe.

     Read three curves per trial, not one. Target val alone cannot separate
     these:
       adaptation      target val down,  ID roughly flat        -- what we want
       specialization  target val down,  ID collapses           -- a trade
       destruction     target val up while train also worsens   -- kill it
     `--lead-cutoff-time` is not the ID probe; step 3's `id_baseline` arm is.

     Kill criterion: val > 1.10x the epoch -1 reference, or two consecutive
     validations worsening. A single small rise is not by itself destructive.

     Read the kill criterion on `val_rmse_K`, NOT on `val_rel_l2`. `val_rel_l2`
     is pooled, sqrt(sum|err|^2 / sum|y|^2), so high-energy pairs dominate it and
     it can fall while most pairs degrade. That is not hypothetical: it is what
     the first N=16 calibration on E11-ca2 did, and it inverted the conclusion.

       LR     pooled val_rel_l2      per-pair val_rmse_K     500-sim scorer
       1e-7   9.7803 -> 9.7860 (ep0)  0.3235 -> 0.3245 (ep0)  --
       3e-7   9.7803 -> 9.8010 (ep0)  0.3235 -> 0.3268 (ep0)  --
       1e-6   9.7803 -> 9.5511 (ep15) 0.3235 -> 0.3384 (ep0)  sinusoid +20%,
                     "best", and wrong                        ID +421%

     Under the pooled metric 1e-6 looks like the one arm that adapts, dipping to
     0.977x at epoch 15. Under the per-pair metric every arm degrades
     monotonically and the best epoch is always 0. The independent scorer agrees
     with the per-pair metric. So `checkpoint_metric` is set to `val_rmse_K`
     here; if you change it, change what you read too.

     Outcome of that calibration, scored on the independent 500-sim arms
     (rel_l2_pct vs the zero-shot checkpoint A; ood_studies/score_lrcal_*):

       LR     sel.ep   sinusoid          id_baseline       uniform_anchor
       --     --       6.9789            1.1404            1.0342
       1e-7   0        6.9803  ( +0.0%)  1.1551  ( +1.3%)  1.0558  ( +2.1%)
       3e-7   0        6.9915  ( +0.2%)  1.2125  ( +6.3%)  1.1377  (+10.0%)
       1e-6   15       8.3924  (+20.3%)  5.9476  (+421%)   5.6184  (+443%)

     There is no adapting regime in that bracket at N=16. LR buys forgetting and
     never buys target accuracy: the OOD arm is flat-to-worse at every LR while
     the ID cost rises monotonically. Full-model FT on 100%-sinusoid batches is
     the mechanism -- 3040 pairs is ~95 unbalanced steps per epoch. Do not read
     the surviving `sinusoid_over_id` improvement as transfer; it shrinks only
     because the denominator collapses. Lowering the LR further approaches a
     no-op (1e-8 already was one), so the next lever is the objective or the
     trainable-parameter set, not the step size.

  2b. Transfer control (--scratch): same split, same normalization, same epoch
     budget, random init. Fine-tuning beating zero-shot only shows the model
     improved; beating this shows the PRETRAINING transferred.

       python scripts/run_ood_finetune.py <ckpt> --scratch --lr 1.75e-4 \
           --data ood_studies/sin_ft_data/data_sinusoid \
           --out ood_studies/sin_scratch_n16 --train-sims 16 --val-sims 16

     Matching --epochs makes this a MATCHED-COMPUTE control, which is the right
     screen but is not the same claim as "scratch cannot learn this family". If
     scratch is still improving at the budget, the honest reading is that
     pretraining cut the optimization and data cost, not that the target is
     unreachable from random init. Before this becomes a headline number, run
     scratch again with a much larger --epochs/--patience and report both.

  3. Score the fine-tuned run with the study script. Symlink the arms the
     source checkpoint was scored on into the new study dir (the ckpt25/ckpt26
     pattern) and pass --reuse-data, so the comparison is the same sims at the
     same --rng-seed and no trajectory is re-solved:

       python scripts/run_ood_spatial_family.py ood_studies/sin_ft_run \
           --out-dir ood_studies/sinusoid_after_ft --reuse-data --rng-seed 7

Step 3 scores the `id_baseline` arm too, which is the forgetting probe: a
sinusoid win that costs in-distribution accuracy is a trade, not a gain.

Normalization is INHERITED from the source checkpoint, never recomputed. The
weights live in model A's normalized space, and a sinusoid arm's own statistics
differ enough (mu 305.5 / sigma 9.06 vs the trained 303.1 / 7.18) that
recomputing would silently redefine the target space. `train.py`'s warm-start
guard enforces exactly this, so the loaders are built here with the
checkpoint's (mu, sigma) pinned and handed to `run_one_seed` as overrides —
which is also what lets this script use the exact locked study cohorts rather
than the generic config split.

Requires a checkpoint trained with spatial_conditioning='spatial_field_only',
for the same reason the zero-shot study does: an unseen family has no honest
8-slot spatial descriptor to put in cond_static.
"""

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ood_spatial_family import (
    _git_commit,
    check_preconditions,
    resolve_checkpoint,
)

DATA_FILE_NAMES = {
    "trajectories.npy": "trajectories.npy",
    "x_grid_path": "x_grid.npy",
    "y_grid_path": "y_grid.npy",
    "t_grid_path": "t_grid.npy",
    "sim_params_path": "sim_params.npy",
}

# Dims the warm-started weights depend on. Checked before any data is touched so
# a spec/checkpoint drift fails at launch with the offending field named, rather
# than as a shape error inside load_state_dict.
DIM_FIELDS = (
    "in_channels", "cond_static_dim", "temporal_token_dim", "s_y_channel",
    "use_temporal_encoder", "use_forcing_time_aug",
    "forcing_extender_rc_cond_index",
)

# A fine-tune arm is tens of sims (~1.2 MB per sim), so it fits in RAM whole;
# reading it eagerly avoids random-access memmap I/O on every pair.
MAX_EAGER_LOAD_BYTES = 2 << 30

# The sim_params field the split balances on. The four temporal forcing families
# differ materially in difficulty, so an unstratified few-shot draw partly
# measures which families happened to land in the subset rather than how much
# target data the model needed.
STRATIFY_KEY = "temporal_family"

LOCKED_COHORT_PLAN = "sinusoid_128_v1"
LOCKED_BASE_TRAIN_IDS = (
    10, 38, 111, 40, 5, 50, 97, 22, 110, 65, 52, 61, 115, 34, 59, 100,
)
LOCKED_VAL_IDS = (
    7, 8, 15, 17, 29, 33, 42, 44, 46, 55, 56, 75, 93, 95, 104, 119,
)
LOCKED_EXPANSION_IDS = (
    92, 31, 80, 27, 90, 62, 88, 32, 26, 87, 125, 60,
    45, 57, 6, 81, 126, 109, 11, 89, 68, 83, 71, 107,
    108, 123, 117, 19, 63, 106, 1, 77, 114, 53, 96, 127,
    9, 16, 43, 67, 64, 35, 37, 74, 47, 72, 86, 122,
)
LOCKED_FINAL_TEST_IDS = (
    39, 2, 118, 82, 14, 58, 99, 21, 18, 102, 101, 103,
    28, 41, 85, 20, 0, 113, 12, 121, 30, 51, 23, 112,
    54, 13, 3, 73, 70, 91, 94, 105, 66, 78, 48, 25,
    4, 24, 76, 120, 84, 36, 98, 69, 49, 79, 116, 124,
)
LOCKED_LABEL_COMPOSITIONS = {
    "base_train": {"exp": 4, "exp_train": 4, "pulse_train": 4, "sin": 4},
    "validation": {"exp": 4, "exp_train": 4, "pulse_train": 4, "sin": 4},
    "expansion": {"exp": 12, "exp_train": 12, "pulse_train": 12, "sin": 12},
    "final_test": {"exp": 12, "exp_train": 18, "pulse_train": 7, "sin": 11},
}


def stratified_order(labels, ids, rng):
    """Order `ids` so every prefix is as label-balanced as `labels` allow.

    Per-label queues are shuffled and then drained round-robin, so a prefix of
    length k holds floor(k / n_labels) or ceil(...) sims of each label until the
    smallest label runs out. That is what makes an 8-sim subset 2+2+2+2 across
    the temporal families instead of a lottery.
    """
    buckets = {}
    for sim_id in ids:
        buckets.setdefault(str(labels[int(sim_id)]), []).append(int(sim_id))
    for label in buckets:
        queue = np.asarray(buckets[label], dtype=np.int64)
        rng.shuffle(queue)
        buckets[label] = queue.tolist()

    ordered = []
    label_names = sorted(buckets)
    while any(buckets[name] for name in label_names):
        for name in label_names:
            if buckets[name]:
                ordered.append(buckets[name].pop(0))
    return np.asarray(ordered, dtype=np.int64)


def resolve_sim_split(labels, train_sims, val_sims, *, seed=0):
    """Stratified, nested train/val/test sim ids for a few-shot transfer sweep.

    Returns (train_ids, val_ids, test_ids).

    Two properties the N-sweep depends on, neither of which a contiguous prefix
    provides:

    * Validation is carved FIRST, so for a fixed (`seed`, `val_sims`) the val
      sims are identical at every `train_sims` and never enter any training
      subset. Sweeping N then moves exactly one variable.
    * Training subsets are NESTED: D_8 subset D_16 subset D_32 subset D_64 for a
      fixed (`seed`, `val_sims`). A change between two N values is additional
      target data, not a differently lucky draw of simulations.

    The test loader is built by `create_dataloaders` and then discarded —
    `run_one_seed` takes only the train and val overrides — so it points at
    whatever sims are left over, or at the val sims when the arm is exactly
    consumed.
    """
    num_sims = len(labels)
    if train_sims < 1 or val_sims < 1:
        raise ValueError("train-sims and val-sims must both be >= 1.")
    if train_sims + val_sims > num_sims:
        raise ValueError(
            f"arm has {num_sims} sims but the split asks for "
            f"{train_sims} train + {val_sims} val. Generate a bigger arm or "
            f"lower the split."
        )
    all_ids = np.arange(num_sims, dtype=np.int64)

    # Two streams, so val selection cannot depend on train_sims and the train
    # ordering cannot depend on how many sims the caller ends up taking.
    val_order = stratified_order(labels, all_ids, np.random.default_rng(seed))
    val_ids = np.sort(val_order[:val_sims])

    remaining = np.setdiff1d(all_ids, val_ids)
    train_order = stratified_order(
        labels, remaining, np.random.default_rng(seed + 1))
    train_ids = train_order[:train_sims]

    rest = train_order[train_sims:]
    return train_ids, val_ids, (np.sort(rest) if rest.size else val_ids)


def label_composition(labels, sim_ids):
    """Count of each stratification label in a split, for the manifest."""
    counts = {}
    for sim_id in sim_ids:
        name = str(labels[int(sim_id)])
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def resolve_locked_study_cohorts(labels, train_sims):
    """Return the immutable D16/D32/D64 transfer-study cohorts.

    D16 uses only ``LOCKED_BASE_TRAIN_IDS``. D32 adds the first 16 IDs in the
    stratified expansion order, and D64 adds all 48. The validation and
    final-test cohorts never change.
    """
    if len(labels) != 128:
        raise ValueError(
            f"{LOCKED_COHORT_PLAN} requires exactly 128 simulations, got {len(labels)}."
        )
    if train_sims not in {16, 32, 64}:
        raise ValueError(
            f"{LOCKED_COHORT_PLAN} supports train-sims 16, 32, or 64, got {train_sims}."
        )

    base = np.asarray(LOCKED_BASE_TRAIN_IDS, dtype=np.int64)
    val = np.asarray(LOCKED_VAL_IDS, dtype=np.int64)
    expansion = np.asarray(LOCKED_EXPANSION_IDS, dtype=np.int64)
    final_test = np.asarray(LOCKED_FINAL_TEST_IDS, dtype=np.int64)
    assigned = np.concatenate([base, val, expansion, final_test])
    if sorted(assigned.tolist()) != list(range(128)):
        raise RuntimeError(f"{LOCKED_COHORT_PLAN} does not partition simulation IDs 0..127.")

    for name, ids in (
        ("base_train", base),
        ("validation", val),
        ("expansion", expansion),
        ("final_test", final_test),
    ):
        expected = LOCKED_LABEL_COMPOSITIONS[name]
        observed = label_composition(labels, ids)
        if observed != expected:
            raise ValueError(
                f"{LOCKED_COHORT_PLAN} {name} labels do not match the locked "
                f"dataset: expected {expected}, got {observed}."
            )

    used_expansion = expansion[:train_sims - len(base)]
    train = base if used_expansion.size == 0 else np.concatenate([base, used_expansion])
    return train, val, expansion, final_test


def assert_dims_match(dims, model_params):
    """Fail when today's ProblemSpec no longer matches the checkpoint's dims."""
    mismatched = {
        field: (model_params[field], getattr(dims, field))
        for field in DIM_FIELDS
        if field in model_params and model_params[field] != getattr(dims, field)
    }
    if mismatched:
        detail = ", ".join(
            f"{k}: checkpoint={ck!r} spec={sp!r}" for k, (ck, sp) in sorted(mismatched.items())
        )
        raise ValueError(
            "checkpoint dims disagree with the resolved ProblemSpec "
            f"({detail}). The warm-started weights cannot load; the benchmark "
            "contract moved since this checkpoint was trained."
        )


def build_finetune_config(source_conf, *, checkpoint_path, data_dir, run_root, args):
    """Derive the fine-tune config from the source checkpoint's own config.

    Starting from the checkpoint's `conf` rather than `conf/config.yaml` keeps
    every architectural knob (forcing_spatial_mode, extender depth,
    spatial_conditioning, ...) exactly as trained, so the only deliberate
    differences are the optimization schedule and where the data comes from.
    """
    config = copy.deepcopy(source_conf)

    source_name = str((config.get("experiment") or {}).get("name", "unknown"))
    if args.scratch:
        suffix = "__scratch_sinusoid"
    elif args.trainable_scope == "boundary_extender":
        suffix = "__ft_extender_sinusoid"
    else:
        suffix = "__ft_sinusoid"
    config["experiment"] = dict(config.get("experiment") or {})
    config["experiment"]["name"] = args.experiment_name or f"{source_name}{suffix}"

    training = config["training"]
    # --scratch keeps the architecture, split, normalization and optimizer budget
    # identical and drops only the pretrained weights. That is the control that
    # turns "fine-tuning helped" into "pretraining transferred": without it, a
    # win over zero-shot could just be N target sims being enough on their own.
    training["init_from_checkpoint"] = (
        None if args.scratch else str(checkpoint_path))
    training["trainable_scope"] = str(args.trainable_scope)
    training["epochs"] = int(args.epochs)
    training["learning_rate"] = float(
        args.extender_lr
        if args.trainable_scope == "boundary_extender" and args.extender_lr is not None
        else args.lr
    )
    training["batch_size"] = int(args.batch_size)
    if args.weight_decay is not None:
        training["weight_decay"] = float(args.weight_decay)
    training["validate_every"] = int(args.validate_every)
    training["patience"] = int(args.patience)
    # Val shares the train snapshot grid (`create_dataloaders` gives the val
    # split `n_snapshots`; `n_snapshots_test` reaches only the test loader, which
    # is discarded here), so selection sees the same lead distribution as
    # training. `n_snapshots_test` is left inherited for later `run_eval.py`.
    training["n_snapshots"] = int(args.n_snapshots)
    # The lead curriculum is a from-scratch device; over a fine-tune this short
    # it would hide the long leads for most of the run.
    training["curriculum_warmup"] = 0
    # NOT val_rel_l2, which is a pooled ratio-of-sums and is therefore an
    # energy-weighted metric: it tracks the highest-signal pairs and can improve
    # while most pairs degrade. The N=16 sinusoid calibration is the worked
    # example -- at 1e-6 the pooled metric read 9.7803 -> 9.5511 and selected
    # epoch 15 as "best", while the per-pair mean over that same val set read
    # 5.8197 -> 8.2430 and the independent 500-sim scorer put the selected
    # checkpoint 20% WORSE on sinusoid and 421% worse in-distribution. val_rmse_K
    # is a per-pair mean and matches the convention every OOD number is reported
    # in, so selection and the headline can no longer disagree in sign.
    training["checkpoint_metric"] = "val_rmse_K"
    # Optional lead diagnostics only. The epoch -1 reference row (model A before
    # any optimizer step, the honest "before" in the training metric convention)
    # is logged by train.py on every warm start and does not depend on this.
    # A --scratch run has no such row, correctly: a random init's "before" is
    # noise, and its comparison point is the fine-tuned run's E_best, not its E_0.
    training["lead_cutoff_time"] = args.lead_cutoff_time
    training["checkpoint_forgetting_ratio"] = None
    training["seeds"] = [int(args.seed)]
    training["num_workers"] = 0
    training["device"] = str(args.device)
    training["run"] = dict(training.get("run") or {})
    training["run"]["run_dir"] = str(run_root)
    if args.constant_lr:
        # gamma=1.0 is how this codebase expresses "no schedule"; there is no
        # ConstantLR branch in build_scheduler and adding one to hold a single
        # value flat would not earn itself.
        training["scheduler"] = {"type": "StepLR", "step_size": 1, "gamma": 1.0}
    # Otherwise the scheduler block is inherited, so RIGNOThreePhase re-derives
    # its phase lengths from the fine-tune `epochs` and its ratios from the new
    # peak LR.

    data = dict(config.get("data") or {})
    for key, file_name in DATA_FILE_NAMES.items():
        data[key] = str(Path(data_dir) / file_name)
    config["data"] = data
    if isinstance(config.get("paths"), dict):
        config["paths"] = dict(config["paths"])
        config["paths"]["data_dir"] = str(data_dir)

    return config


def check_schedule_fits(config):
    """Pre-empt the trainer's phase-length error while --epochs is still fixable.

    The inherited RIGNOThreePhase block derives its phase lengths from
    `training.epochs`, and rounds warmup and exp up to 1 epoch each, so a very
    short fine-tune leaves the cosine phase empty. Raised here rather than 30
    seconds later inside `build_scheduler`.

    Returns the resolved phase lengths so the caller can print them. The block
    was tuned for a 36-60 epoch from-scratch run; re-derived over a 20-epoch
    warm start it can spend a surprising share of the run in the warmup and
    exponential tails, and that is worth seeing before the run, not after.
    """
    scheduler = config["training"].get("scheduler") or {}
    if (str(scheduler.get("type")) == "StepLR"
            and float(scheduler.get("gamma", 0.0)) == 1.0):
        return {
            "type": "constant",
            "constant_lr": float(config["training"]["learning_rate"]),
        }
    if str(scheduler.get("type")) != "RIGNOThreePhase":
        return None
    from src.operators.train import (
        _resolve_rigno_lr_values,
        _resolve_rigno_phase_lengths,
    )

    try:
        warmup, cosine, exp = _resolve_rigno_phase_lengths(
            config["training"], scheduler)
    except ValueError as exc:
        raise SystemExit(
            f"--epochs {config['training']['epochs']} is too short for the "
            f"inherited RIGNOThreePhase schedule ({exc})."
        ) from exc
    total = int(config["training"]["epochs"])
    described = {
        "type": "RIGNOThreePhase",
        "warmup_epochs": warmup,
        "cosine_epochs": cosine,
        "exp_epochs": exp,
        "transition_fraction_of_run": round((warmup + exp) / total, 3),
    }
    try:
        peak_lr, init_lr, cosine_floor_lr, final_lr = _resolve_rigno_lr_values(
            config["training"], scheduler)
    except ValueError:
        # The LR values are diagnostic only; the trainer raises its own error on
        # a malformed block. Losing them must not cost the phase-length check.
        return described
    described.update(init_lr=init_lr, peak_lr=peak_lr,
                     cosine_floor_lr=cosine_floor_lr, final_lr=final_lr)
    return described


def inherited_provenance(checkpoint, checkpoint_path):
    """Model A's normalization provenance, tagged as inherited by this run.

    `run_one_seed` reads this off the train dataset when loaders are overridden
    and bakes it into the fine-tuned checkpoint, so downstream eval still proves
    which training population defined the normalization — model A's, unchanged.
    """
    source = checkpoint.get("normalization_provenance")
    if not source:
        return None
    payload = dict(source)
    payload["inherited_from_checkpoint"] = str(checkpoint_path)
    payload["inherited_reason"] = (
        "fine-tune reuses the source checkpoint's (mu_global, sigma_global); "
        "the fine-tune arm's own statistics are deliberately NOT used."
    )
    return payload


def default_device():
    import torch

    # resolve_device('auto') only ever picks CUDA or CPU, so a local fine-tune
    # would silently land on CPU (~10x slower here) without this.
    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return "mps"
    return "auto"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint", help="Path to a run_root or a bare fno2d_best.pt")
    p.add_argument("--data", required=True,
                   help="Fine-tune arm directory (e.g. <study>/data_sinusoid)")
    p.add_argument("--out", required=True,
                   help="Run root; training writes <out>/seed<seed>/")
    p.add_argument("--source-seed", type=int, default=None,
                   help="Seed subdir when the checkpoint is a run_root")
    p.add_argument("--seed", type=int, default=42, help="Fine-tune seed")
    p.add_argument("--train-sims", type=int, choices=(16, 32, 64), default=16,
                   help="Locked target-data condition: D16 uses the base cohort; "
                        "D32 adds the first 16 expansion simulations; D64 adds "
                        "all 48 expansion simulations.")
    p.add_argument("--val-sims", type=int, default=16)
    p.add_argument("--split-seed", type=int, default=0,
                   help="Recorded for compatibility; the study split is locked "
                        "to seed 0 and immutable simulation IDs.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-7,
                   help="Peak LR. This is the LEAST-DAMAGE setting, not a "
                        "known-good one: the completed N=16 calibration on "
                        "E11-ca2 (see the module docstring) found no LR in "
                        "[1e-7, 1e-6] that improves the sinusoid arm, and 3e-7 "
                        "already costs 6-10% in-distribution for +0.2% on "
                        "target. 1e-7 is nearly a no-op (+1.3% ID) and 1e-6 is "
                        "destructive (+421% ID). Recalibrate (--constant-lr, "
                        "N=16) for a different checkpoint, N, or batch size.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=None,
                   help="Default: inherit the source checkpoint's value")
    p.add_argument("--n-snapshots", type=int, default=20,
                   help="Snapshots subsampled per sim; all-to-all pairs over "
                        "them, for train and val alike (20 -> 190 pairs/sim, "
                        "31 -> 465)")
    p.add_argument("--validate-every", type=int, default=2)
    p.add_argument("--lead-cutoff-time", type=float, default=None,
                   help="Adds pre/post-cutoff lead diagnostics split at this "
                        "lead. Does not change checkpoint selection, and is NOT "
                        "required for the epoch -1 reference row: train.py logs "
                        "that on every warm start.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Build config, split and loaders, report, and stop")
    p.add_argument("--constant-lr", action="store_true",
                   help="Hold --lr flat for the whole run instead of "
                        "inheriting the source schedule. Use for LR "
                        "calibration: a decaying schedule confounds 'this "
                        "update scale is safe' with 'the decay arrived before "
                        "the damage did'.")
    p.add_argument(
        "--trainable-scope", choices=("full", "boundary_extender"), default="full",
        help="Update every parameter, or freeze the model except for the existing "
             "boundary extender module.",
    )
    p.add_argument(
        "--extender-lr", type=float, default=None,
        help="Peak LR used only with --trainable-scope boundary_extender. If "
             "omitted, --lr is used.",
    )
    p.add_argument("--scratch", action="store_true",
                   help="Transfer control: train a random init on the SAME "
                        "target sims, split, normalization and epoch budget. "
                        "The checkpoint is still read, for its architecture and "
                        "its (mu, sigma) only. Requires an explicit --lr.")
    args = p.parse_args(argv)
    if args.device is None:
        args.device = default_device()
    if args.val_sims != 16 or args.split_seed != 0:
        p.error(
            "this study uses the locked sinusoid_128_v1 cohorts: "
            "--val-sims must be 16 and --split-seed must be 0."
        )
    if args.extender_lr is not None and args.trainable_scope != "boundary_extender":
        p.error("--extender-lr requires --trainable-scope boundary_extender.")
    if args.scratch and args.trainable_scope != "full":
        p.error("--scratch is the all-weights control and requires --trainable-scope full.")
    if args.scratch and "--lr" not in (argv if argv is not None else sys.argv[1:]):
        # The warm-start default is ~500x below the from-scratch peak, because
        # it is calibrated not to disturb pretrained weights. Handing it to a
        # random init would build a strawman baseline and manufacture positive
        # transfer, so make the choice deliberate.
        p.error("--scratch requires an explicit --lr: the warm-start default "
                "is orders of magnitude below a from-scratch peak and would "
                "understate the baseline. The source run used 2.8e-3 at batch "
                "512 (~1.75e-4 at batch 32).")
    return args


def main(argv=None):
    args = parse_args(argv)

    data_dir = Path(args.data).expanduser()
    run_root = Path(args.out).expanduser()
    run_dir = run_root / f"seed{int(args.seed)}"

    checkpoint_path, checkpoint, source_seed_name = resolve_checkpoint(
        args.checkpoint, args.source_seed)
    check_preconditions(checkpoint)

    config = build_finetune_config(
        checkpoint["conf"], checkpoint_path=checkpoint_path, data_dir=data_dir,
        run_root=run_root, args=args,
    )

    from data.dataset import (
        create_dataloaders,
        load_ramp_seconds,
        load_sim_data,
        load_solver_dt,
        problem_from_config,
    )
    from src.operators.train import run_one_seed

    spec = problem_from_config(config)
    assert_dims_match(spec.dims, checkpoint["conf"]["model"]["parameters"])
    schedule = check_schedule_fits(config)

    mu_global = float(checkpoint["mu_global"])
    sigma_global = float(checkpoint["sigma_global"])

    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    if trajectories.nbytes <= MAX_EAGER_LOAD_BYTES:
        trajectories = np.asarray(trajectories)
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    solver_dt = load_solver_dt(config["data"]["t_grid_path"])
    ramp_seconds = load_ramp_seconds(config["data"]["t_grid_path"])

    strat_labels = [
        str(sim_params[i].get(STRATIFY_KEY, "?"))
        for i in range(int(trajectories.shape[0]))
    ]
    train_ids, val_ids, expansion_ids, final_test_ids = resolve_locked_study_cohorts(
        strat_labels, int(args.train_sims),
    )
    spec.validate_schema(sim_params, np.concatenate([train_ids, val_ids]))

    train_loader, val_loader, _ = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=final_test_ids,
        batch_size=int(args.batch_size),
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=int(args.n_snapshots),
        num_workers=0,
        dt=solver_dt,
        ramp_seconds=ramp_seconds,
        temporal_samples=config["model"]["parameters"].get("temporal_samples", 64),
        sampler_seed=int(args.seed),
        problem=spec,
    )
    train_loader.dataset.normalization_provenance = inherited_provenance(
        checkpoint, checkpoint_path)

    families = sorted({
        str(sim_params[int(i)].get("spatial_family", "?")) for i in train_ids
    })

    run_root.mkdir(parents=True, exist_ok=True)
    if args.scratch:
        method = "scratch_baseline"
    elif args.trainable_scope == "boundary_extender":
        method = "warm_start_boundary_extender_finetune"
    else:
        method = "warm_start_full_finetune"
    effective_lr = float(config["training"]["learning_rate"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "method": method,
        "source_checkpoint": {
            "weights_loaded": not args.scratch,
            "path": str(checkpoint_path),
            "seed_dir": source_seed_name,
            "experiment": checkpoint["conf"].get("experiment", {}).get("name"),
            "epoch": checkpoint.get("epoch"),
            "best_val": checkpoint.get("best_val"),
            "spatial_conditioning": checkpoint["conf"]["benchmark"].get(
                "spatial_conditioning"),
            "forcing_spatial_mode": checkpoint["conf"]["model"]["parameters"].get(
                "forcing_spatial_mode"),
            "forcing_cond_mode": checkpoint["conf"]["model"]["parameters"].get(
                "forcing_cond_mode"),
            "forcing_extender_condition_on_rc": checkpoint["conf"]["model"][
                "parameters"
            ].get("forcing_extender_condition_on_rc", False),
            "forcing_extender_physics_hidden": checkpoint["conf"]["model"][
                "parameters"
            ].get("forcing_extender_physics_hidden", 16),
            "forcing_extender_interface_x_norm": checkpoint["conf"]["model"][
                "parameters"
            ].get("forcing_extender_interface_x_norm", 0.5),
        },
        "finetune_data": {
            "data_dir": str(data_dir),
            "num_sims_available": int(trajectories.shape[0]),
            # The shot count is SIMULATIONS. `train_pairs` is an all-to-all
            # expansion over snapshots of those same trajectories, so it is not
            # a count of independent target examples and must not be reported
            # as one.
            "n_shot_train_sims": int(len(train_ids)),
            "n_val_sims": int(len(val_ids)),
            "train_sim_ids": train_ids.tolist(),
            "val_sim_ids": val_ids.tolist(),
            "base_train_sim_ids": list(LOCKED_BASE_TRAIN_IDS),
            "expansion_sim_ids": expansion_ids.tolist(),
            "used_expansion_sim_ids": train_ids[len(LOCKED_BASE_TRAIN_IDS):].tolist(),
            "unused_expansion_sim_ids": expansion_ids[
                len(train_ids) - len(LOCKED_BASE_TRAIN_IDS):
            ].tolist(),
            "final_test_sim_ids": final_test_ids.tolist(),
            "spatial_families_in_train": families,
            "n_snapshots": int(args.n_snapshots),
            "train_pairs": len(train_loader.dataset),
            "val_pairs": len(val_loader.dataset),
        },
        "split": {
            "cohort_plan": LOCKED_COHORT_PLAN,
            "stratify_key": STRATIFY_KEY,
            "split_seed": int(args.split_seed),
            "scheme": "immutable 16 base-train / 16 validation / 48 expansion / "
                      "48 final-test simulation IDs; D32 = base-train + first 16 "
                      "expansion IDs; D64 = base-train + all expansion IDs",
            "train_label_composition": label_composition(strat_labels, train_ids),
            "val_label_composition": label_composition(strat_labels, val_ids),
            "expansion_label_composition": label_composition(
                strat_labels, expansion_ids),
            "final_test_label_composition": label_composition(
                strat_labels, final_test_ids),
        },
        "normalization": {
            "mu_global": mu_global,
            "sigma_global": sigma_global,
            "source": "inherited from the source checkpoint (never recomputed)",
        },
        "optimization": {
            "epochs": int(args.epochs),
            "learning_rate": effective_lr,
            "full_model_lr_argument": float(args.lr),
            "extender_lr_argument": (
                None if args.extender_lr is None else float(args.extender_lr)
            ),
            "trainable_scope": str(args.trainable_scope),
            "batch_size": int(args.batch_size),
            "weight_decay": config["training"]["weight_decay"],
            "scheduler": config["training"].get("scheduler", {}).get("type"),
            "lr_schedule": schedule,
            "curriculum_warmup": 0,
            "device": str(args.device),
            "seed": int(args.seed),
        },
        "run_dir": str(run_dir),
    }
    (run_root / "finetune_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("=" * 78)
    if args.scratch:
        print("OOD SCRATCH BASELINE (random init, all weights trainable)")
    elif args.trainable_scope == "boundary_extender":
        print("OOD FINE-TUNE (warm start, boundary extender only)")
    else:
        print("OOD FINE-TUNE (warm start, all weights trainable)")
    print("=" * 78)
    print(f"  source      : {checkpoint_path}"
          f"{'  [architecture + (mu,sigma) only]' if args.scratch else ''}")
    print(f"                {manifest['source_checkpoint']['experiment']} "
          f"(epoch {checkpoint.get('epoch')}, best_val "
          f"{checkpoint.get('best_val')})")
    print(f"  arm         : {data_dir} ({trajectories.shape[0]} sims available)")
    print(f"  train       : {len(train_ids)} sims -> "
          f"{len(train_loader.dataset)} pairs  (families: {', '.join(families)})")
    print(f"  val         : {len(val_ids)} sims -> {len(val_loader.dataset)} pairs")
    print(f"  norm        : mu={mu_global:.6f} sigma={sigma_global:.6f} (inherited)")
    print(f"  split       : {LOCKED_COHORT_PLAN}; immutable IDs, stratified on "
          f"{STRATIFY_KEY}")
    print(f"                train {manifest['split']['train_label_composition']}")
    print(f"                val   {manifest['split']['val_label_composition']}")
    used_expansion = len(train_ids) - len(LOCKED_BASE_TRAIN_IDS)
    print(f"  expansion   : {used_expansion}/{len(expansion_ids)} sims used; "
          f"{len(expansion_ids) - used_expansion} remain unused")
    print(f"  final test  : {len(final_test_ids)} sims untouched")
    print(f"  optim       : {args.epochs} epochs, {args.trainable_scope}, "
          f"peak lr {effective_lr:g}, batch {args.batch_size}, device {args.device}")
    if schedule is not None and schedule["type"] == "constant":
        print(f"  lr curve    : constant {schedule['constant_lr']:g} for all "
              f"{args.epochs} epochs (calibration mode; no decay to hide "
              f"behind)")
    elif schedule is not None:
        if "peak_lr" in schedule:
            print(f"  lr curve    : {schedule['init_lr']:g} -> "
                  f"{schedule['peak_lr']:g} -> {schedule['cosine_floor_lr']:g} -> "
                  f"{schedule['final_lr']:g}")
        print(f"                warmup {schedule['warmup_epochs']}ep / cosine "
              f"{schedule['cosine_epochs']}ep / exp {schedule['exp_epochs']}ep "
              f"({schedule['transition_fraction_of_run']:.0%} of the run is "
              f"warmup+exp tails)")
    print(f"  run_dir     : {run_dir}")
    print(f"Wrote {run_root / 'finetune_manifest.json'}", flush=True)

    if args.dry_run:
        return 0

    result = run_one_seed(
        config, int(args.seed), run_dir=run_dir,
        train_loader_override=train_loader, val_loader_override=val_loader,
    )
    print(json.dumps(result, indent=2))
    print()
    print("Score it against the arms the source checkpoint was scored on:")
    print(f"  python scripts/run_ood_spatial_family.py {run_root} \\")
    print(f"      --out-dir <study_dir> --reuse-data --rng-seed 7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
