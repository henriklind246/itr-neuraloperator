"""Publication tables built from provenance-linked pair-level test records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from visual.pub import records
from visual.pub.manifest import FigureSource, ProvenanceError, sha256_file

BENCH_ORDER = ("forcing", "interfaces", "source", "source_itr")
BENCH_LABEL = {
    "forcing": "Forcing",
    "interfaces": "Interfaces",
    "source": "Source",
    "source_itr": "Source + ITR",
}
N_BOOT = 10_000
BOOTSTRAP_RNG_SEED = 20260728
EXPECTED_N_SIMS = 150
EXPECTED_PAIRS_PER_SIM = 465
EXPECTED_INITIAL_TARGETS = 30

PRIMARY_SPECS = (
    ("field_rmse_mean_K", "Field RMSE mean (K)"),
    ("field_rmse_p95_K", "Field RMSE P95 (K)"),
    ("field_gnrmse_pct", "Field GNRMSE (%)"),
    ("jump_rmse_mean_K", "Jump RMSE mean (K)"),
    ("jump_rmse_p95_K", "Jump RMSE P95 (K)"),
    ("peak_jump_error_mean_K", "Peak-jump error mean (K)"),
    ("peak_jump_error_p95_K", "Peak-jump error P95 (K)"),
)

DESCRIPTIVE_SPECS = PRIMARY_SPECS + (
    ("field_pooled_rmse_K", "Field pooled RMSE (K)"),
    (
        "field_centered_pooled_rel_l2_pct",
        "Field training-centered pooled rel-L2 (%)",
    ),
    ("interface_pooled_rmse_K", "Interface-band pooled RMSE (K)"),
    (
        "interface_centered_pooled_rel_l2_pct",
        "Interface-band training-centered pooled rel-L2 (%)",
    ),
    ("signed_peak_bias_K", "Signed peak bias (K)"),
)


def _p95(values) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95,
                             method="linear"))


def _lead_step(frame: pd.DataFrame) -> pd.Series:
    return frame["j"].astype(np.int64) - frame["s"].astype(np.int64)


def _cluster_dot(counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum("bi,i->b", counts, values, optimize=True)


def _peak_frame(frame: pd.DataFrame) -> pd.DataFrame:
    initial = frame[(frame["s"] == 0) & (frame["j"] > 0)].copy()
    counts = initial.groupby("sim_id").size()
    if counts.empty or counts.nunique() != 1:
        raise ProvenanceError(
            "initial-state peak histories are incomplete or inconsistent"
        )
    grouped = initial.groupby("sim_id", sort=True)
    out = grouped[["node_jump_abs_max_pred_K", "node_jump_abs_max_true_K"]].max()
    out["signed_peak_bias_K"] = (
        out["node_jump_abs_max_pred_K"] - out["node_jump_abs_max_true_K"]
    )
    out["peak_jump_error_K"] = np.abs(out["signed_peak_bias_K"])
    return out.reset_index()


def _seed_estimates(frame: pd.DataFrame, rise_scale_K: float) -> dict[str, float]:
    peak = _peak_frame(frame)
    pooled_rmse = float(np.sqrt(frame["sse_K2"].sum()
                                / frame["num_error_cells"].sum()))
    interface_pooled_rmse = float(np.sqrt(
        frame["interface_sse_K2"].sum() / frame["num_interface_cells"].sum()
    ))
    return {
        "field_rmse_mean_K": float(frame["rmse_K"].mean()),
        "field_rmse_p95_K": _p95(frame["rmse_K"]),
        "field_gnrmse_pct": 100.0 * pooled_rmse / float(rise_scale_K),
        "field_pooled_rmse_K": pooled_rmse,
        "field_centered_pooled_rel_l2_pct": 100.0 * float(np.sqrt(
            frame["sse_K2"].sum() / frame["target_sse_K2"].sum()
        )),
        "interface_pooled_rmse_K": interface_pooled_rmse,
        "interface_centered_pooled_rel_l2_pct": 100.0 * float(np.sqrt(
            frame["interface_sse_K2"].sum()
            / frame["interface_target_sse_K2"].sum()
        )),
        "jump_rmse_mean_K": float(frame["node_jump_rmse_K"].mean()),
        "jump_rmse_p95_K": _p95(frame["node_jump_rmse_K"]),
        "peak_jump_error_mean_K": float(peak["peak_jump_error_K"].mean()),
        "peak_jump_error_p95_K": _p95(peak["peak_jump_error_K"]),
        "signed_peak_bias_K": float(peak["signed_peak_bias_K"].mean()),
    }


def _weighted_linear_quantile(
    sorted_values: np.ndarray,
    sorted_sim_codes: np.ndarray,
    cluster_counts: np.ndarray,
    q: float,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Exact NumPy-linear quantile of cluster-resampled observations."""
    n_boot = cluster_counts.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    observations_per_sim = np.bincount(
        sorted_sim_codes, minlength=cluster_counts.shape[1]
    ).astype(np.int64)
    total = _cluster_dot(cluster_counts, observations_per_sim)
    h = (total - 1) * float(q)
    lo_pos = np.floor(h).astype(np.int64)
    hi_pos = np.ceil(h).astype(np.int64)
    frac = h - lo_pos
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        weights = cluster_counts[start:stop, sorted_sim_codes]
        cumulative = np.cumsum(weights, axis=1)
        lo_idx = np.argmax(cumulative > lo_pos[start:stop, None], axis=1)
        hi_idx = np.argmax(cumulative > hi_pos[start:stop, None], axis=1)
        lo = sorted_values[lo_idx]
        hi = sorted_values[hi_idx]
        out[start:stop] = lo + frac[start:stop] * (hi - lo)
    return out


def _cluster_bootstrap_seed(
    frame: pd.DataFrame,
    rise_scale_K: float,
    cluster_counts: np.ndarray,
    sim_order: np.ndarray,
) -> dict[str, np.ndarray]:
    cluster_counts_f = cluster_counts.astype(np.float64, copy=False)
    sim_to_code = {int(s): i for i, s in enumerate(sim_order)}
    codes = frame["sim_id"].map(sim_to_code).to_numpy(dtype=np.int64)
    n_pairs_by_sim = np.bincount(codes, minlength=len(sim_order)).astype(np.float64)

    field = frame["rmse_K"].to_numpy(dtype=np.float64)
    jump = frame["node_jump_rmse_K"].to_numpy(dtype=np.float64)
    field_sum = np.bincount(codes, weights=field, minlength=len(sim_order))
    jump_sum = np.bincount(codes, weights=jump, minlength=len(sim_order))
    denom = _cluster_dot(cluster_counts_f, n_pairs_by_sim)

    result = {
        "field_rmse_mean_K": _cluster_dot(cluster_counts_f, field_sum) / denom,
        "jump_rmse_mean_K": _cluster_dot(cluster_counts_f, jump_sum) / denom,
    }
    for key, values in (
        ("field_rmse_p95_K", field), ("jump_rmse_p95_K", jump)
    ):
        order = np.argsort(values, kind="mergesort")
        result[key] = _weighted_linear_quantile(
            values[order], codes[order], cluster_counts, 0.95
        )

    sse = np.bincount(
        codes, weights=frame["sse_K2"].to_numpy(dtype=np.float64),
        minlength=len(sim_order),
    )
    cells = np.bincount(
        codes, weights=frame["num_error_cells"].to_numpy(dtype=np.float64),
        minlength=len(sim_order),
    )
    result["field_gnrmse_pct"] = (
        100.0 * np.sqrt(_cluster_dot(cluster_counts_f, sse)
                        / _cluster_dot(cluster_counts_f, cells))
        / float(rise_scale_K)
    )
    result["field_pooled_rmse_K"] = np.sqrt(
        _cluster_dot(cluster_counts_f, sse)
        / _cluster_dot(cluster_counts_f, cells)
    )
    target_sse = np.bincount(
        codes, weights=frame["target_sse_K2"].to_numpy(dtype=np.float64),
        minlength=len(sim_order),
    )
    result["field_centered_pooled_rel_l2_pct"] = 100.0 * np.sqrt(
        _cluster_dot(cluster_counts_f, sse)
        / _cluster_dot(cluster_counts_f, target_sse)
    )
    interface_sse = np.bincount(
        codes, weights=frame["interface_sse_K2"].to_numpy(dtype=np.float64),
        minlength=len(sim_order),
    )
    interface_cells = np.bincount(
        codes, weights=frame["num_interface_cells"].to_numpy(dtype=np.float64),
        minlength=len(sim_order),
    )
    interface_target_sse = np.bincount(
        codes,
        weights=frame["interface_target_sse_K2"].to_numpy(dtype=np.float64),
        minlength=len(sim_order),
    )
    result["interface_pooled_rmse_K"] = np.sqrt(
        _cluster_dot(cluster_counts_f, interface_sse)
        / _cluster_dot(cluster_counts_f, interface_cells)
    )
    result["interface_centered_pooled_rel_l2_pct"] = 100.0 * np.sqrt(
        _cluster_dot(cluster_counts_f, interface_sse)
        / _cluster_dot(cluster_counts_f, interface_target_sse)
    )

    peak = _peak_frame(frame).set_index("sim_id").loc[sim_order]
    for key, column in (
        ("peak_jump_error_mean_K", "peak_jump_error_K"),
        ("signed_peak_bias_K", "signed_peak_bias_K"),
    ):
        values = peak[column].to_numpy(dtype=np.float64)
        result[key] = _cluster_dot(cluster_counts_f, values) / cluster_counts_f.sum(axis=1)
    peak_values = peak["peak_jump_error_K"].to_numpy(dtype=np.float64)
    order = np.argsort(peak_values, kind="mergesort")
    result["peak_jump_error_p95_K"] = _weighted_linear_quantile(
        peak_values[order], order.astype(np.int64), cluster_counts, 0.95
    )
    return result


def _lead_balanced_seed(frame: pd.DataFrame, rise_scale_K: float) -> dict[str, float]:
    per_lead = []
    for _, part in frame.groupby(_lead_step(frame), sort=True):
        pooled = float(np.sqrt(part["sse_K2"].sum()
                               / part["num_error_cells"].sum()))
        per_lead.append({
            "field_rmse_mean_K": float(part["rmse_K"].mean()),
            "field_rmse_p95_K": _p95(part["rmse_K"]),
            "field_gnrmse_pct": 100.0 * pooled / float(rise_scale_K),
            "jump_rmse_mean_K": float(part["node_jump_rmse_K"].mean()),
            "jump_rmse_p95_K": _p95(part["node_jump_rmse_K"]),
        })
    if not per_lead:
        raise ProvenanceError("records contain no positive lead times")
    return {
        f"lead_balanced_{key}": float(np.mean([row[key] for row in per_lead]))
        for key in per_lead[0]
    }


def _lead_balanced_bootstrap_seed(
    frame: pd.DataFrame,
    rise_scale_K: float,
    cluster_counts: np.ndarray,
    sim_order: np.ndarray,
) -> dict[str, np.ndarray]:
    cluster_counts_f = cluster_counts.astype(np.float64, copy=False)
    sim_to_code = {int(s): i for i, s in enumerate(sim_order)}
    collected: dict[str, list[np.ndarray]] = {}
    for _, part in frame.groupby(_lead_step(frame), sort=True):
        codes = part["sim_id"].map(sim_to_code).to_numpy(dtype=np.int64)
        n_by_sim = np.bincount(codes, minlength=len(sim_order)).astype(np.float64)
        denom = _cluster_dot(cluster_counts_f, n_by_sim)
        for name, column in (
            ("field_rmse", "rmse_K"), ("jump_rmse", "node_jump_rmse_K")
        ):
            values = part[column].to_numpy(dtype=np.float64)
            sums = np.bincount(
                codes, weights=values, minlength=len(sim_order)
            ).astype(np.float64)
            collected.setdefault(f"{name}_mean_K", []).append(
                _cluster_dot(cluster_counts_f, sums) / denom
            )
            order = np.argsort(values, kind="mergesort")
            collected.setdefault(f"{name}_p95_K", []).append(
                _weighted_linear_quantile(
                    values[order], codes[order], cluster_counts, 0.95
                )
            )
        sse = np.bincount(
            codes, weights=part["sse_K2"].to_numpy(dtype=np.float64),
            minlength=len(sim_order),
        )
        cells = np.bincount(
            codes, weights=part["num_error_cells"].to_numpy(dtype=np.float64),
            minlength=len(sim_order),
        )
        collected.setdefault("field_gnrmse_pct", []).append(
            100.0 * np.sqrt(
                _cluster_dot(cluster_counts_f, sse)
                / _cluster_dot(cluster_counts_f, cells)
            )
            / float(rise_scale_K)
        )
    return {
        f"lead_balanced_{key}": np.mean(np.stack(values), axis=0)
        for key, values in collected.items()
    }


def _validate_seed_frames(
    benchmark: str,
    seed_frames: dict[str, pd.DataFrame],
    provenance: dict[str, dict],
    *,
    strict_counts: bool,
    min_seeds: int = 3,
    required_representation: str | None = "temporal_encoder",
) -> tuple[np.ndarray, dict[str, float]]:
    if len(seed_frames) < min_seeds:
        raise ProvenanceError(
            f"{benchmark}: at least {min_seeds} predeclared seeds required"
        )
    hashes = {
        (p.get("training_population_hash"), p.get("normalization_definition_hash"))
        for p in provenance.values()
    }
    if len(hashes) != 1 or any(value is None for pair in hashes for value in pair):
        raise ProvenanceError(
            f"{benchmark}: training-population/normalization hashes are missing or differ"
        )
    evaluation_hashes = {p.get("evaluation_population_hash") for p in provenance.values()}
    if len(evaluation_hashes) != 1 or None in evaluation_hashes:
        raise ProvenanceError(
            f"{benchmark}: evaluation-population hashes are missing or differ"
        )

    scales = {}
    means, sigmas = [], []
    key_reference = None
    sim_reference = None
    for seed, frame in seed_frames.items():
        p = provenance[seed]
        if p.get("prediction_mode") != "direct_pair":
            raise ProvenanceError(f"{benchmark} seed {seed}: direct_pair records required")
        if p.get("benchmark") != benchmark:
            raise ProvenanceError(f"{benchmark} seed {seed}: provenance benchmark mismatch")
        if (required_representation is not None
                and p.get("representation") != required_representation):
            raise ProvenanceError(
                f"{benchmark} seed {seed}: {required_representation} "
                "representation required"
            )
        mu = float(p["training_mean_K"])
        sigma = float(p["training_population_std_K"])
        scale = float(p["training_temperature_rise_rms_K"])
        expected = float(np.sqrt(sigma ** 2 + (mu - float(p["temperature_reference_K"])) ** 2))
        if not np.isclose(scale, expected, rtol=1e-12, atol=1e-12):
            raise ProvenanceError(f"{benchmark} seed {seed}: invalid training rise scale")
        means.append(mu)
        sigmas.append(sigma)
        scales[seed] = scale

        keys = frame[["sim_id", "s", "j"]].sort_values(
            ["sim_id", "s", "j"]
        ).to_records(index=False).tolist()
        sims = np.sort(frame["sim_id"].unique())
        key_reference = keys if key_reference is None else key_reference
        sim_reference = sims if sim_reference is None else sim_reference
        if keys != key_reference or not np.array_equal(sims, sim_reference):
            raise ProvenanceError(f"{benchmark}: seeds do not share identical evaluation pairs")
        counts = frame.groupby("sim_id").size()
        initial_counts = frame[(frame["s"] == 0) & (frame["j"] > 0)].groupby("sim_id").size()
        if strict_counts and (
            len(sims) != EXPECTED_N_SIMS
            or set(counts.to_numpy()) != {EXPECTED_PAIRS_PER_SIM}
            or set(initial_counts.to_numpy()) != {EXPECTED_INITIAL_TARGETS}
        ):
            raise ProvenanceError(
                f"{benchmark} seed {seed}: expected {EXPECTED_N_SIMS} simulations, "
                f"{EXPECTED_PAIRS_PER_SIM} pairs/simulation and "
                f"{EXPECTED_INITIAL_TARGETS} initial-state targets"
            )

    if not np.allclose(means, means[0], rtol=1e-6, atol=1e-8) or not np.allclose(
        sigmas, sigmas[0], rtol=1e-6, atol=1e-8
    ):
        raise ProvenanceError(
            f"{benchmark}: identical normalization populations produced inconsistent scales"
        )
    return np.asarray(sim_reference, dtype=np.int64), scales


def _artifact_groups(source: FigureSource):
    grouped: dict[str, list[Path]] = {}
    for artifact in source.artifacts:
        if artifact.kind != "test_records":
            continue
        for benchmark in artifact.benchmarks:
            grouped.setdefault(benchmark, []).append(Path(artifact.path))
    return grouped


def write_primary_results(
    source: FigureSource,
    out_dir: str | Path,
    *,
    n_boot: int = N_BOOT,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
    strict_counts: bool = True,
) -> tuple[Path, Path, Path]:
    """Write compact, detailed, and provenance artifacts for table T01."""
    groups = _artifact_groups(source)
    missing = [benchmark for benchmark in BENCH_ORDER if benchmark not in groups]
    if missing:
        raise ProvenanceError(f"T01 missing benchmark records: {missing}")

    compact_rows = []
    detail_rows = []
    source_paths = []
    rng = np.random.default_rng(rng_seed)

    for benchmark in BENCH_ORDER:
        seed_frames, provenance = {}, {}
        for path in groups[benchmark]:
            try:
                record = records.load_test_records(path, require_version=4)
                run_provenance = records.load_test_record_provenance(path)
            except (records.SchemaError, FileNotFoundError) as exc:
                raise ProvenanceError(f"{benchmark}: {exc}") from exc
            if record.schema_version < 4:
                raise ProvenanceError(f"{path}: schema v4 required for T01")
            seed = str(record.seed)
            if seed in seed_frames:
                raise ProvenanceError(f"{benchmark}: duplicate records for seed {seed}")
            frame = record.df[record.df["benchmark"] == benchmark].copy()
            seed_frames[seed] = frame
            provenance[seed] = run_provenance
            source_paths.extend([path, records.provenance_path_for(path)])

        sim_order, scales = _validate_seed_frames(
            benchmark, seed_frames, provenance, strict_counts=strict_counts
        )
        seed_estimates = {
            seed: _seed_estimates(frame, scales[seed])
            for seed, frame in seed_frames.items()
        }
        seed_lead = {
            seed: _lead_balanced_seed(frame, scales[seed])
            for seed, frame in seed_frames.items()
        }

        sampled = rng.integers(0, len(sim_order), size=(n_boot, len(sim_order)))
        cluster_counts = np.zeros((n_boot, len(sim_order)), dtype=np.int16)
        row_idx = np.repeat(np.arange(n_boot), len(sim_order))
        np.add.at(cluster_counts, (row_idx, sampled.reshape(-1)), 1)
        boot_by_seed = {
            seed: _cluster_bootstrap_seed(frame, scales[seed], cluster_counts, sim_order)
            for seed, frame in seed_frames.items()
        }
        lead_boot_by_seed = {
            seed: _lead_balanced_bootstrap_seed(
                frame, scales[seed], cluster_counts, sim_order
            )
            for seed, frame in seed_frames.items()
        }

        compact = {"Benchmark": BENCH_LABEL[benchmark]}
        all_metrics = list(PRIMARY_SPECS) + [("signed_peak_bias_K", "Signed peak bias (K)")]
        for metric, label in all_metrics:
            values = np.array(
                [seed_estimates[seed][metric] for seed in sorted(seed_frames)],
                dtype=np.float64,
            )
            estimate = float(values.mean())
            boot = np.mean(
                np.stack([boot_by_seed[seed][metric] for seed in sorted(seed_frames)]),
                axis=0,
            )
            lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
            if metric in dict(PRIMARY_SPECS):
                compact[dict(PRIMARY_SPECS)[metric]] = estimate
            detail_rows.append({
                "benchmark": BENCH_LABEL[benchmark],
                "aggregation_level": "across_seed",
                "seed": "",
                "metric": metric,
                "statistic": label,
                "estimate": estimate,
                "test_cluster_ci_lower": float(lo),
                "test_cluster_ci_upper": float(hi),
                "across_seed_sd": float(np.std(values, ddof=1)),
                "n_seeds": len(seed_frames),
                "n_simulations": len(sim_order),
                "n_pairs": len(next(iter(seed_frames.values()))),
                "estimand": (
                    "uniform valid source-target pair"
                    if "peak" not in metric else "initial-state-conditioned trajectory"
                ),
                "evidence_directional_peak_bias": bool(
                    metric == "signed_peak_bias_K" and (lo > 0.0 or hi < 0.0)
                ),
            })
            for seed in sorted(seed_frames):
                detail_rows.append({
                    "benchmark": BENCH_LABEL[benchmark],
                    "aggregation_level": "seed",
                    "seed": seed,
                    "metric": metric,
                    "statistic": label,
                    "estimate": seed_estimates[seed][metric],
                    "test_cluster_ci_lower": np.nan,
                    "test_cluster_ci_upper": np.nan,
                    "across_seed_sd": np.nan,
                    "n_seeds": 1,
                    "n_simulations": len(sim_order),
                    "n_pairs": len(seed_frames[seed]),
                    "estimand": "seed-specific descriptive estimate",
                    "evidence_directional_peak_bias": "",
                })

        for metric in next(iter(seed_lead.values())):
            values = np.array([seed_lead[s][metric] for s in sorted(seed_lead)])
            boot = np.mean(
                np.stack([
                    lead_boot_by_seed[s][metric] for s in sorted(seed_lead)
                ]), axis=0,
            )
            lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
            n_leads = int(_lead_step(next(iter(seed_frames.values()))).nunique())
            detail_rows.append({
                "benchmark": BENCH_LABEL[benchmark],
                "aggregation_level": "lead_balanced_across_seed",
                "seed": "",
                "metric": metric,
                "statistic": f"Equal-weight mean over {n_leads} lead-specific statistics",
                "estimate": float(values.mean()),
                "test_cluster_ci_lower": float(lo),
                "test_cluster_ci_upper": float(hi),
                "across_seed_sd": float(np.std(values, ddof=1)),
                "n_seeds": len(seed_frames),
                "n_simulations": len(sim_order),
                "n_pairs": len(next(iter(seed_frames.values()))),
                "estimand": "uniform positive lead time sensitivity",
                "evidence_directional_peak_bias": "",
            })
        compact_rows.append(compact)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_path = out_dir / "T01_primary_results.csv"
    detail_path = out_dir / "T01_primary_results_details.csv"
    provenance_path = out_dir / "T01_primary_results.provenance.json"
    pd.DataFrame(compact_rows).to_csv(compact_path, index=False)
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    provenance_payload = {
        "schema": "visual.pub.table-provenance/1",
        "table_key": "T01_primary_results",
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(set(source_paths))
        ],
        "bootstrap": {
            "method": "simulation-cluster percentile",
            "n_replicates": int(n_boot),
            "rng_seed": int(rng_seed),
            "interpretation": (
                "Finite held-out test-cohort uncertainty conditional on the "
                "fixed trained models; not total training-and-evaluation uncertainty."
            ),
        },
        "seed_aggregation": (
            "Arithmetic mean of seed-specific statistics; P95 is the mean of "
            "seed-specific P95 values, not a percentile of pooled seed predictions."
        ),
        "pair_estimand": (
            "Performance for a uniformly selected valid source-target pair; "
            "short lead times consequently receive more weight."
        ),
        "caption": (
            "Pairwise metrics are calculated over 69,750 source-target pairs "
            "from 150 held-out simulations per benchmark and model seed. "
            "Peak-jump metrics are calculated over 150 initial-state-conditioned "
            "reconstructed trajectories. Each entry is the arithmetic mean of "
            "the corresponding seed-level statistic across the predeclared seeds."
        ),
        "failed_run_policy": {
            "infrastructure_invalid": "rerun the same declared seed/configuration",
            "completed_poor": "retain in the publication cohort",
            "invalid_checkpoint": "exclude until the same declared seed is validly rerun",
            "representation_incompatible": "replace only by documented same-seed/config rerun",
        },
    }
    provenance_path.write_text(json.dumps(provenance_payload, indent=2))
    return compact_path, detail_path, provenance_path


def write_descriptive_results(
    source: FigureSource,
    out_dir: str | Path,
    *,
    n_boot: int = N_BOOT,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
    strict_counts: bool = True,
) -> tuple[Path, Path, Path]:
    """Write the explicitly single-checkpoint, mixed-representation T01 preview."""
    groups = _artifact_groups(source)
    missing = [benchmark for benchmark in BENCH_ORDER if benchmark not in groups]
    if missing:
        raise ProvenanceError(f"descriptive T01 missing benchmark records: {missing}")

    compact_rows = []
    detail_rows = []
    source_paths = []
    cohort = []
    rng = np.random.default_rng(rng_seed)

    for benchmark in BENCH_ORDER:
        paths = groups[benchmark]
        if len(paths) != 1:
            raise ProvenanceError(
                f"{benchmark}: descriptive T01 requires exactly one checkpoint "
                f"record file, found {len(paths)}"
            )
        path = paths[0]
        try:
            record = records.load_test_records(path, require_version=4)
            run_provenance = records.load_test_record_provenance(path)
        except (records.SchemaError, FileNotFoundError) as exc:
            raise ProvenanceError(f"{benchmark}: {exc}") from exc
        if record.schema_version < 4:
            raise ProvenanceError(f"{path}: schema v4 required for descriptive T01")
        seed = str(record.seed)
        frame = record.df[record.df["benchmark"] == benchmark].copy()
        if frame.empty:
            raise ProvenanceError(f"{path}: no {benchmark!r} records")
        sim_order, scales = _validate_seed_frames(
            benchmark,
            {seed: frame},
            {seed: run_provenance},
            strict_counts=strict_counts,
            min_seeds=1,
            required_representation=None,
        )
        representation = str(run_provenance.get("representation"))
        if representation not in {"temporal_encoder", "bins"}:
            raise ProvenanceError(
                f"{benchmark} seed {seed}: unsupported representation "
                f"{representation!r}"
            )
        estimate = _seed_estimates(frame, scales[seed])
        lead_estimate = _lead_balanced_seed(frame, scales[seed])

        sampled = rng.integers(0, len(sim_order), size=(n_boot, len(sim_order)))
        cluster_counts = np.zeros((n_boot, len(sim_order)), dtype=np.int16)
        np.add.at(
            cluster_counts,
            (
                np.repeat(np.arange(n_boot), len(sim_order)),
                sampled.reshape(-1),
            ),
            1,
        )
        boot = _cluster_bootstrap_seed(
            frame, scales[seed], cluster_counts, sim_order
        )
        lead_boot = _lead_balanced_bootstrap_seed(
            frame, scales[seed], cluster_counts, sim_order
        )

        compact = {
            "Benchmark": BENCH_LABEL[benchmark],
            "Representation": representation,
            "Seed": seed,
        }
        for metric, label in DESCRIPTIVE_SPECS:
            lo, hi = np.quantile(boot[metric], [0.025, 0.975], method="linear")
            compact[label] = estimate[metric]
            detail_rows.append({
                "benchmark": BENCH_LABEL[benchmark],
                "representation": representation,
                "aggregation_level": "single_checkpoint",
                "seed": seed,
                "metric": metric,
                "statistic": label,
                "estimate": estimate[metric],
                "test_cluster_ci_lower": float(lo),
                "test_cluster_ci_upper": float(hi),
                "across_seed_sd": np.nan,
                "n_seeds": 1,
                "n_simulations": len(sim_order),
                "n_pairs": len(frame),
                "estimand": (
                    "uniform valid source-target pair"
                    if "peak" not in metric
                    else "initial-state-conditioned trajectory"
                ),
                "evidence_directional_peak_bias": bool(
                    metric == "signed_peak_bias_K" and (lo > 0.0 or hi < 0.0)
                ),
            })
        n_leads = int(_lead_step(frame).nunique())
        for metric, value in lead_estimate.items():
            lo, hi = np.quantile(
                lead_boot[metric], [0.025, 0.975], method="linear"
            )
            detail_rows.append({
                "benchmark": BENCH_LABEL[benchmark],
                "representation": representation,
                "aggregation_level": "lead_balanced_single_checkpoint",
                "seed": seed,
                "metric": metric,
                "statistic": (
                    f"Equal-weight mean over {n_leads} lead-specific statistics"
                ),
                "estimate": value,
                "test_cluster_ci_lower": float(lo),
                "test_cluster_ci_upper": float(hi),
                "across_seed_sd": np.nan,
                "n_seeds": 1,
                "n_simulations": len(sim_order),
                "n_pairs": len(frame),
                "estimand": "uniform positive lead time sensitivity",
                "evidence_directional_peak_bias": "",
            })
        compact_rows.append(compact)
        source_paths.extend([path, records.provenance_path_for(path)])
        cohort.append({
            "benchmark": benchmark,
            "representation": representation,
            "seed": seed,
            "checkpoint_sha256": run_provenance.get("checkpoint_sha256"),
            "training_population_hash": run_provenance.get(
                "training_population_hash"
            ),
            "normalization_definition_hash": run_provenance.get(
                "normalization_definition_hash"
            ),
            "evaluation_population_hash": run_provenance.get(
                "evaluation_population_hash"
            ),
            "normalization_provenance_source": run_provenance.get(
                "normalization_provenance_source"
            ),
            "training_population_identity_status": (
                "checkpoint_embedded"
                if run_provenance.get("normalization_provenance_source") == "checkpoint"
                else "reconstructed_sidecar_for_legacy_checkpoint"
            ),
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_path = out_dir / "T01_primary_results_descriptive.csv"
    detail_path = out_dir / "T01_primary_results_descriptive_details.csv"
    provenance_path = out_dir / "T01_primary_results_descriptive.provenance.json"
    pd.DataFrame(compact_rows).to_csv(compact_path, index=False)
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    representations = {row["representation"] for row in cohort}
    provenance_payload = {
        "schema": "visual.pub.table-provenance/1",
        "table_key": "T01_primary_results_descriptive",
        "publication_readiness": "descriptive_only",
        "strict_table_unchanged": True,
        "strict_cohort_blockers": [
            "one checkpoint per benchmark; at least three predeclared seeds are required",
            "source_itr uses bins; the strict cohort requires temporal_encoder",
        ],
        "mixed_representation": len(representations) > 1,
        "cohort": cohort,
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(set(source_paths))
        ],
        "bootstrap": {
            "method": "simulation-cluster percentile",
            "n_replicates": int(n_boot),
            "rng_seed": int(rng_seed),
            "interpretation": (
                "Finite held-out test-cohort uncertainty conditional on each fixed "
                "trained checkpoint; it excludes training-seed uncertainty."
            ),
        },
        "pair_estimand": (
            "Performance for a uniformly selected valid source-target pair; "
            "short lead times consequently receive more weight."
        ),
        "gnrmse_definition": (
            "100 * sqrt(sum physical-field SSE / sum field cells) / "
            "sqrt(training population variance + "
            "(training population mean - 300 K)^2)"
        ),
        "caption": (
            "Descriptive fixed-checkpoint evaluation over 69,750 source-target "
            "pairs from 150 held-out simulations per benchmark. Conditional "
            "simulation-cluster 95% intervals do not quantify training-seed "
            "variability. Source + ITR temporarily uses the bins representation."
        ),
    }
    provenance_path.write_text(json.dumps(provenance_payload, indent=2))
    return compact_path, detail_path, provenance_path


def lead_time_curve(
    frame: pd.DataFrame,
    metric: str,
    *,
    n_boot: int = N_BOOT,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> pd.DataFrame:
    """Exact-lead seed-averaged mean/P95 with a simulation-cluster mean CI."""
    column = {"field": "rmse_K", "jump": "node_jump_rmse_K"}.get(metric)
    if column is None:
        raise KeyError("metric must be 'field' or 'jump'")
    seeds = sorted(frame["seed"].astype(str).unique())
    sim_order = np.sort(frame["sim_id"].unique())
    rng = np.random.default_rng(rng_seed)
    sampled = rng.integers(0, len(sim_order), size=(n_boot, len(sim_order)))
    counts = np.zeros((n_boot, len(sim_order)), dtype=np.int16)
    np.add.at(
        counts,
        (np.repeat(np.arange(n_boot), len(sim_order)), sampled.reshape(-1)),
        1,
    )
    sim_to_code = {int(s): i for i, s in enumerate(sim_order)}
    counts_f = counts.astype(np.float64, copy=False)
    lead_steps = np.sort(_lead_step(frame).unique())
    rows = []
    for lead_step in lead_steps:
        means, p95s, boot_means = [], [], []
        n_pairs = 0
        for seed in seeds:
            part = frame[
                (frame["seed"].astype(str) == seed)
                & (_lead_step(frame) == lead_step)
            ]
            values = part[column].to_numpy(dtype=np.float64)
            codes = part["sim_id"].map(sim_to_code).to_numpy(dtype=np.int64)
            per_sum = np.bincount(codes, weights=values, minlength=len(sim_order))
            per_n = np.bincount(codes, minlength=len(sim_order)).astype(np.float64)
            means.append(float(values.mean()))
            p95s.append(_p95(values))
            boot_means.append(
                _cluster_dot(counts_f, per_sum) / _cluster_dot(counts_f, per_n)
            )
            n_pairs = len(part)
        boot = np.mean(np.stack(boot_means), axis=0)
        lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
        rows.append({
            "t_bar": float(frame.loc[_lead_step(frame) == lead_step, "t_bar"].mean()),
            "mean": float(np.mean(means)),
            "p95": float(np.mean(p95s)),
            "test_cluster_ci_lower": float(lo),
            "test_cluster_ci_upper": float(hi),
            "n_pairs_per_seed": int(n_pairs),
            "n_simulations": int(len(sim_order)),
            "n_seeds": int(len(seeds)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "BENCH_ORDER", "BENCH_LABEL", "PRIMARY_SPECS", "DESCRIPTIVE_SPECS",
    "write_primary_results", "write_descriptive_results", "lead_time_curve",
]
