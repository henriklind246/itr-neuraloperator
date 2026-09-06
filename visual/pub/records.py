"""Loading and schema-versioning for the CSV artifacts the paper figures consume.

This module loads and validates. It never aggregates and never plots: every
reduction lives in ``visual/pub/stats.py``.

Schema versions of ``test_records.csv``:

- v1: the original 26 columns. Carries per-pair ``rmse_K``/``rel_l2_pct`` only.
  A v1 frame cannot be pooled correctly to the simulation level, because the
  per-cell squared-error sums were never written. Loading one is allowed but
  always attaches ``SCHEMA_V1_NO_POOLED_STATS``.
- v2: adds the pooled sufficient statistics (``sse_K2`` etc.). This is the
  minimum for any quantitative figure in the paper.
- v3: adds the protocol / OOD identity columns needed by the OOD figures.
- v4: links every row to run-level provenance and stores the per-target maximum
  absolute predicted/truth node jump needed for initial-state peak histories.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.operators.eval import TEST_RECORD_FIELDS
from visual.pub.manifest import Degradation

# The 26 columns actually present in the surviving v1 artifact
# (runs/source_itr_smoke3/config0/seed42/test_records.csv).
SCHEMA_V1_COLUMNS: tuple[str, ...] = (
    "sim_id", "s", "j", "t_s", "t_bar", "R_c", "benchmark",
    "temporal_family", "spatial_family",
    "x_h", "y_h", "A", "freq", "regime",
    "R_c_amp", "R_c_y0", "R_c_sigma",
    "x_I", "rel_l2_pct", "iface_rel_l2_pct",
    "nrmse_pct", "rmse_K", "gnrmse_pct",
    "node_jump_rmse_K", "node_jump_nrmse_pct", "node_jump_gnrmse_pct",
)

# Pooled sufficient statistics. Without these, sim-level aggregation is a mean
# of per-pair RMSEs, which is a different (and wrong) statistic.
SCHEMA_V2_REQUIRED: tuple[str, ...] = (
    "sse_K2", "num_error_cells", "target_sse_K2",
    "interface_sse_K2", "num_interface_cells", "interface_target_sse_K2",
)

SCHEMA_V3_REQUIRED: tuple[str, ...] = (
    "protocol", "distribution_class", "ood_axis",
    "lead_time_actual", "time_norm_horizon",
)

SCHEMA_V4_REQUIRED: tuple[str, ...] = (
    "provenance_id", "node_jump_abs_max_pred_K", "node_jump_abs_max_true_K",
)

_INT_COLS = ("sim_id", "s", "j")
_STR_COLS = (
    "provenance_id", "benchmark", "temporal_family", "spatial_family", "regime",
    "protocol", "distribution_class", "ood_axis", "ood_value", "latents_hash",
)


def detect_schema_version(columns) -> int:
    """Return 4, 3, 2, or 1 for a ``test_records`` column set.

    Raises when the frame is not even a v1 test-records table, so a mis-pointed
    manifest entry fails at load rather than producing an empty figure.
    """
    cols = set(columns)
    missing_v1 = [c for c in SCHEMA_V1_COLUMNS if c not in cols]
    if missing_v1:
        raise SchemaError(
            "Not a test_records table; missing v1 columns: "
            f"{missing_v1[:6]}{'...' if len(missing_v1) > 6 else ''}"
        )
    if all(c in cols for c in SCHEMA_V2_REQUIRED):
        if all(c in cols for c in SCHEMA_V3_REQUIRED) and all(
            c in cols for c in SCHEMA_V4_REQUIRED
        ):
            return 4
        if all(c in cols for c in SCHEMA_V3_REQUIRED):
            return 3
        return 2
    return 1


class SchemaError(ValueError):
    """Raised when an artifact does not match any known schema version."""


@dataclass
class RecordFrame:
    """A loaded ``test_records.csv`` with its schema version and any degradations."""

    df: pd.DataFrame
    schema_version: int
    path: Path
    seed: str
    degradations: list[Degradation] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return int(len(self.df))

    @property
    def n_simulations(self) -> int:
        return int(self.df["sim_id"].nunique()) if "sim_id" in self.df.columns else 0

    @property
    def benchmarks(self) -> tuple[str, ...]:
        if "benchmark" not in self.df.columns:
            return ()
        return tuple(sorted(self.df["benchmark"].dropna().unique().tolist()))

    @property
    def has_pooled_stats(self) -> bool:
        return self.schema_version >= 2


@dataclass(frozen=True)
class RolloutArmRecords:
    """One direct or autoregressive arm with its linked run provenance."""

    benchmark: str
    representation: str
    seed: str
    substeps: int
    prediction_mode: str
    records: RecordFrame
    provenance: dict


@dataclass(frozen=True)
class ResolutionStudy:
    """Reported fixed-checkpoint errors over one resolution ladder."""

    label: str
    benchmark: str
    material_side: bool
    seed: str
    checkpoint_epoch: int
    resolutions: tuple[int, ...]
    global_rel_l2_pct: tuple[float, ...]
    interface_rel_l2_pct: tuple[float, ...]
    boundary_rel_l2_pct: tuple[float, ...]
    n_simulations: int
    pairs_per_simulation: int
    comparison_caveat: str


@dataclass(frozen=True)
class ResolutionDrift:
    """Metric-matched FV discretization drift to the finest grid."""

    resolutions: tuple[int, ...]
    global_rel_l2_pct: tuple[float, ...]
    interface_rel_l2_pct: tuple[float, ...]
    boundary_rel_l2_pct: tuple[float, ...]
    reference_resolution: int


@dataclass(frozen=True)
class ResolutionStudies:
    """The original/material-side checkpoint comparison and FV reference."""

    original: ResolutionStudy
    material_side: ResolutionStudy
    fv_drift: ResolutionDrift


def seed_from_path(csv_path: Path) -> str:
    """Return the seed tag from a ``.../seed42/test_records.csv`` style path."""
    name = Path(csv_path).parent.name
    if name.startswith("seed"):
        return name[len("seed"):]
    return name


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in df.columns:
        if col in _STR_COLS:
            df[col] = df[col].fillna("").astype(str)
        elif col not in _INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_test_records(path: str | Path, *, require_version: int = 2,
                      seed: str | None = None) -> RecordFrame:
    """Load one ``test_records.csv``.

    ``require_version`` is a soft floor: a lower version still loads, but the
    returned frame carries a ``Degradation`` so every downstream summary is
    forced to declare that its numbers are not pooled correctly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"test_records not found: {path}")
    df = pd.read_csv(path)
    version = detect_schema_version(df.columns)
    df = _coerce(df)

    degradations: list[Degradation] = []
    if version < require_version:
        if version == 1:
            degradations.append(Degradation(
                "SCHEMA_V1_NO_POOLED_STATS",
                f"{path} is schema v1: pooled sufficient statistics "
                f"({', '.join(SCHEMA_V2_REQUIRED)}) are absent, so per-simulation "
                "metrics fall back to an unpooled mean of per-pair values. "
                "Re-run scripts/write_test_records.py to obtain v2.",
            ))
        else:
            degradations.append(Degradation(
                "SCHEMA_BELOW_REQUIRED",
                f"{path} is schema v{version}; figure requires v{require_version}.",
            ))

    resolved_seed = seed if seed is not None else seed_from_path(path)
    if "seed" not in df.columns or df["seed"].isna().all():
        df["seed"] = resolved_seed
    df["seed"] = df["seed"].astype(str)

    return RecordFrame(df=df, schema_version=version, path=path,
                       seed=resolved_seed, degradations=degradations)


def provenance_path_for(records_path: str | Path) -> Path:
    path = Path(records_path)
    return path.with_name(f"{path.stem}.provenance.json")


def load_test_record_provenance(records_path: str | Path) -> dict:
    """Load and verify the run-level metadata linked by schema-v4 records."""
    records_path = Path(records_path)
    provenance_path = provenance_path_for(records_path)
    if not provenance_path.exists():
        raise SchemaError(f"test-record provenance not found: {provenance_path}")
    payload = json.loads(provenance_path.read_text())
    if payload.get("schema") != "test-records-provenance/v1":
        raise SchemaError(
            f"{provenance_path} has unsupported schema {payload.get('schema')!r}"
        )
    header = pd.read_csv(records_path, usecols=["provenance_id"])
    ids = set(header["provenance_id"].dropna().astype(str).unique())
    expected = str(payload.get("provenance_id", ""))
    if ids != {expected}:
        raise SchemaError(
            f"{records_path} provenance IDs {sorted(ids)} do not match "
            f"{provenance_path} ({expected!r})"
        )
    return payload


def load_many_test_records(paths, *, require_version: int = 2) -> RecordFrame:
    """Concatenate several per-seed record files into one frame.

    The lowest schema version present governs, so a single stale seed cannot
    silently upgrade the whole set.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("load_many_test_records requires at least one path")
    frames = [load_test_records(p, require_version=require_version) for p in paths]
    version = min(f.schema_version for f in frames)
    degradations: list[Degradation] = []
    for f in frames:
        degradations.extend(f.degradations)
    df = pd.concat([f.df for f in frames], ignore_index=True)
    seeds = sorted({f.seed for f in frames})
    return RecordFrame(df=df, schema_version=version, path=paths[0],
                       seed=",".join(seeds), degradations=degradations)


def load_train_metrics(path: str | Path) -> pd.DataFrame:
    """Load a ``train_metrics.csv`` and tag it with its seed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"train_metrics not found: {path}")
    df = pd.read_csv(path)
    if "seed" not in df.columns:
        df["seed"] = seed_from_path(path)
    df["seed"] = df["seed"].astype(str)
    return df


def load_csv_table(path: str | Path, *, required_columns: tuple[str, ...] = ()
                   ) -> pd.DataFrame:
    """Load a generic tabular artifact (inverse results, OOD per-sim rollups).

    Missing required columns raise rather than yielding an empty or partial
    figure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing required columns: {missing}")
    return df


def count_simulations(df: pd.DataFrame) -> int:
    """Number of distinct simulations, i.e. the true replication count."""
    return int(df["sim_id"].nunique()) if "sim_id" in df.columns else 0


def available_strata(df: pd.DataFrame, candidates) -> tuple[str, ...]:
    """Return the candidate stratum columns that are present and non-degenerate."""
    out: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col]
        if values.isna().all():
            continue
        if values.dtype == object and (values.astype(str).str.strip() == "").all():
            continue
        if values.nunique(dropna=True) < 2:
            continue
        out.append(col)
    return tuple(out)


def artifact_paths(source, kind: str, *, benchmark: str | None = None
                   ) -> list[Path]:
    """Paths of one artifact kind on a resolved ``FigureSource``."""
    out: list[Path] = []
    for ref in getattr(source, "artifacts", ()) or ():
        if ref.kind != kind:
            continue
        if benchmark is not None and ref.benchmarks and benchmark not in ref.benchmarks:
            continue
        out.append(Path(ref.path))
    return out


def records_by_benchmark(source, *, require_version: int = 2
                         ) -> dict[str, RecordFrame]:
    """Every ``test_records`` artifact of a source, grouped by benchmark.

    Seeds of the same benchmark are concatenated into one frame, so downstream
    aggregation sees the seed as an ordinary grouping column rather than as
    separate inputs that a figure would have to stitch together itself.
    """
    by_bench: dict[str, list[Path]] = {}
    for ref in getattr(source, "artifacts", ()) or ():
        if ref.kind != "test_records":
            continue
        for bench in (ref.benchmarks or ("unknown",)):
            by_bench.setdefault(str(bench), []).append(Path(ref.path))

    out: dict[str, RecordFrame] = {}
    for bench, paths in sorted(by_bench.items()):
        frame = load_many_test_records(paths, require_version=require_version)
        if bench != "unknown" and "benchmark" in frame.df.columns:
            keep = frame.df["benchmark"].astype(str) == bench
            frame.df = frame.df[keep].reset_index(drop=True)
        out[bench] = frame
    return out


INVERSE_SENSOR_COUNTS = (8, 16, 32)
INVERSE_SENSOR_BENCHMARKS = ("forcing", "forcing_itr")
_INVERSE_SENSOR_BASE_COLUMNS = (
    "benchmark", "sim_id", "n_sensors", "noise_seed", "init_seed",
    "noise_std_K", "fv_resid_rms_K", "fv_resid_over_noise",
    "profile_bound_limited",
)
_INVERSE_SENSOR_COLUMNS = {
    "forcing": (
        "R_c_true", "R_c_map", "R_c_abs_error",
        "profile_R_c_ci_low", "profile_R_c_ci_high",
    ),
    "forcing_itr": (
        "excess_int_true", "excess_int_hat", "excess_int_abserr",
        "profile_excess_ci_low", "profile_excess_ci_high",
    ),
}


def load_inverse_sensor_sweep(
    source,
    *,
    sensor_counts: tuple[int, ...] = INVERSE_SENSOR_COUNTS,
    benchmarks: tuple[str, ...] = INVERSE_SENSOR_BENCHMARKS,
    n_cases: int = 8,
) -> pd.DataFrame:
    """Load the paired per-case inverse sweep used by F18.

    One table per benchmark or one combined table are both accepted. Every
    benchmark must contain the same case/noise/initialization keys at each of
    the three discrete sensor counts. This function validates table identity
    and shape only; numerical reductions remain in ``visual.pub.stats``.
    """
    paths = artifact_paths(source, "inverse_csv")
    if not paths:
        raise SchemaError("inverse sensor sweep source contains no CSV artifacts")

    frames = [
        load_csv_table(path, required_columns=_INVERSE_SENSOR_BASE_COLUMNS)
        for path in paths
    ]
    table = pd.concat(frames, ignore_index=True, sort=False)
    table["benchmark"] = table["benchmark"].fillna("").astype(str)

    unexpected = sorted(set(table["benchmark"]) - set(benchmarks))
    if unexpected:
        raise SchemaError(
            f"inverse sensor sweep contains unsupported benchmarks {unexpected}"
        )
    missing_benchmarks = [
        benchmark for benchmark in benchmarks
        if not (table["benchmark"] == benchmark).any()
    ]
    if missing_benchmarks:
        raise SchemaError(
            f"inverse sensor sweep is missing benchmarks {missing_benchmarks}"
        )

    numeric_base = (
        "sim_id", "n_sensors", "noise_seed", "init_seed", "noise_std_K",
        "fv_resid_rms_K", "fv_resid_over_noise",
    )
    for column in numeric_base:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[list(numeric_base)].isna().any().any():
        raise SchemaError("inverse sensor sweep has non-numeric pairing or FV fields")

    bool_values = table["profile_bound_limited"].map(
        {True: True, False: False, 1: True, 0: False,
         "True": True, "False": False, "true": True, "false": False,
         "1": True, "0": False, "1.0": True, "0.0": False}
    )
    if bool_values.isna().any():
        raise SchemaError("inverse sensor sweep has invalid profile_bound_limited values")
    table["profile_bound_limited"] = bool_values.astype(bool)

    expected_counts = tuple(sensor_counts)
    pairing_columns = ["sim_id", "noise_seed", "init_seed"]
    for benchmark in benchmarks:
        part = table.loc[table["benchmark"] == benchmark]
        conditional = _INVERSE_SENSOR_COLUMNS[benchmark]
        missing = [column for column in conditional if column not in part.columns]
        if missing:
            raise SchemaError(
                f"inverse sensor sweep for {benchmark} is missing columns {missing}"
            )
        for column in conditional:
            table.loc[part.index, column] = pd.to_numeric(
                part[column], errors="coerce"
            )
        if table.loc[part.index, list(conditional)].isna().any().any():
            raise SchemaError(
                f"inverse sensor sweep for {benchmark} has non-numeric metric fields"
            )

        counts = tuple(sorted(part["n_sensors"].astype(int).unique()))
        if counts != expected_counts:
            raise SchemaError(
                f"inverse sensor sweep for {benchmark} has sensor counts {counts}; "
                f"expected {expected_counts}"
            )
        if part.duplicated(["n_sensors", *pairing_columns]).any():
            raise SchemaError(
                f"inverse sensor sweep for {benchmark} contains duplicate paired rows"
            )

        keys_by_count = {
            count: {
                tuple(int(value) for value in row)
                for row in part.loc[
                    part["n_sensors"].astype(int) == count, pairing_columns
                ].itertuples(index=False, name=None)
            }
            for count in expected_counts
        }
        reference = keys_by_count[expected_counts[0]]
        if len(reference) != n_cases or any(
            keys != reference for keys in keys_by_count.values()
        ):
            raise SchemaError(
                f"inverse sensor sweep for {benchmark} requires the same "
                f"{n_cases} (sim_id, noise_seed, init_seed) cases at every "
                f"sensor count; got {keys_by_count}"
            )

    return table.sort_values(
        ["benchmark", "sim_id", "noise_seed", "init_seed", "n_sensors"]
    ).reset_index(drop=True)


def load_rollout_arms(source, *, require_version: int = 4
                      ) -> dict[str, dict[int, RolloutArmRecords]]:
    """Load and validate pair-aligned direct/K rollout record sets.

    Arm identity comes from each schema-v4 provenance sidecar rather than its
    filename. Within a benchmark, every arm must use the same checkpoint and
    evaluation population, and its raw ``(sim_id, s, j)`` keys must match the
    direct arm exactly.
    """
    paths = artifact_paths(source, "test_records")
    if not paths:
        raise SchemaError("rollout source contains no test_records artifacts")

    by_benchmark: dict[str, dict[int, RolloutArmRecords]] = {}
    pair_keys: dict[tuple[str, int], set[tuple[int, int, int]]] = {}
    for path in paths:
        frame = load_test_records(path, require_version=require_version)
        if frame.schema_version < require_version:
            raise SchemaError(
                f"{path} is schema v{frame.schema_version}; rollout comparison "
                f"requires v{require_version}"
            )
        provenance = load_test_record_provenance(path)
        benchmark = str(provenance.get("benchmark", ""))
        representation = str(provenance.get("representation", ""))
        seed = str(provenance.get("seed", frame.seed))
        prediction_mode = str(provenance.get("prediction_mode", ""))
        substeps = int(provenance.get("rollout_num_substeps", 0))
        enabled = bool(provenance.get("rollout_enabled", False))

        if not benchmark or representation != "temporal_encoder":
            raise SchemaError(
                f"{path} has unsupported rollout provenance: "
                f"benchmark={benchmark!r}, representation={representation!r}"
            )
        direct = prediction_mode == "direct_pair" and not enabled and substeps == 1
        autoregressive = (
            prediction_mode == f"autoregressive_{substeps}_substeps"
            and enabled and substeps > 1
        )
        if not (direct or autoregressive):
            raise SchemaError(
                f"{path} has inconsistent rollout mode {prediction_mode!r}, "
                f"enabled={enabled}, substeps={substeps}"
            )

        arms = by_benchmark.setdefault(benchmark, {})
        if substeps in arms:
            raise SchemaError(
                f"duplicate {benchmark} rollout arm with {substeps} substeps"
            )
        arms[substeps] = RolloutArmRecords(
            benchmark=benchmark,
            representation=representation,
            seed=seed,
            substeps=substeps,
            prediction_mode=prediction_mode,
            records=frame,
            provenance=provenance,
        )
        pair_keys[(benchmark, substeps)] = {
            (int(sim_id), int(s), int(j))
            for sim_id, s, j in frame.df[["sim_id", "s", "j"]].itertuples(
                index=False, name=None
            )
        }

    for benchmark, arms in by_benchmark.items():
        if set(arms) != {1, 2, 4, 8}:
            raise SchemaError(
                f"{benchmark} rollout arms are {sorted(arms)}; expected [1, 2, 4, 8]"
            )
        invariant_fields = (
            "evaluation_population_hash", "checkpoint_sha256", "seed",
            "representation", "rollout_partition",
        )
        for field_name in invariant_fields:
            values = {str(arm.provenance.get(field_name)) for arm in arms.values()}
            if len(values) != 1 or values <= {"", "None"}:
                raise SchemaError(
                    f"{benchmark} rollout arms disagree on {field_name}: "
                    f"{sorted(values)}"
                )
        direct_keys = pair_keys[(benchmark, 1)]
        for substeps in (2, 4, 8):
            if pair_keys[(benchmark, substeps)] != direct_keys:
                raise SchemaError(
                    f"{benchmark} K={substeps} does not contain the direct arm's "
                    "exact (sim_id, s, j) evaluation population"
                )

    return dict(sorted(by_benchmark.items()))


_RESOLUTION_REPORT_RE = re.compile(r"seed_report_r(?P<resolution>\d+)\.json$")


def _json_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"JSON report must contain an object: {path}")
    return payload


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_resolution_studies(source) -> ResolutionStudies:
    """Load E32, E33 material-side, and the normalized FV drift ladder."""
    paths = artifact_paths(source, "json_report")
    manifest_paths = [p for p in paths if p.name == "study_manifest.json"]
    drift_paths = [p for p in paths if p.name == "fv_drift_baseline.json"]
    if len(manifest_paths) != 2:
        raise SchemaError(
            f"resolution source needs two study manifests, found {len(manifest_paths)}"
        )
    if len(drift_paths) != 1:
        raise SchemaError(
            f"resolution source needs one metric-matched FV drift report, "
            f"found {len(drift_paths)}"
        )

    studies: list[ResolutionStudy] = []
    for manifest_path in manifest_paths:
        study_root = manifest_path.parent
        manifest = _json_payload(manifest_path)
        expected_resolutions = tuple(int(v) for v in manifest.get("resolutions", ()))
        reports: dict[int, dict] = {}
        for path in paths:
            match = _RESOLUTION_REPORT_RE.match(path.name)
            if match and _under(path, study_root):
                reports[int(match.group("resolution"))] = _json_payload(path)
        if tuple(sorted(reports)) != expected_resolutions:
            raise SchemaError(
                f"{manifest_path} declares {expected_resolutions}, but its report "
                f"set is {tuple(sorted(reports))}"
            )

        per_resolution: list[dict] = []
        for resolution in expected_resolutions:
            per_seed = reports[resolution].get("per_seed") or []
            if len(per_seed) != 1:
                raise SchemaError(
                    f"{study_root} r{resolution} must report exactly one fixed seed"
                )
            per_resolution.append(per_seed[0])

        seeds = {str(row.get("seed")) for row in per_resolution}
        epochs = {int(row.get("best_epoch")) for row in per_resolution}
        sim_counts = {int(row.get("num_sims")) for row in per_resolution}
        if len(seeds) != 1 or len(epochs) != 1 or len(sim_counts) != 1:
            raise SchemaError(
                f"{study_root} changes seed, checkpoint epoch, or simulation "
                "count across resolutions"
            )
        benchmark = str(manifest.get("benchmark", ""))
        if benchmark != "forcing":
            raise SchemaError(
                f"{manifest_path} is benchmark {benchmark!r}; expected 'forcing'"
            )

        material_side = bool(manifest.get("material_side", False))
        pairs_per_sim = int(manifest.get("pairs_per_sim", 0))
        if not pairs_per_sim:
            validations = [
                _json_payload(p) for p in paths
                if p.name == "validation.json" and _under(p, study_root)
            ]
            if len(validations) == 1:
                pairs_per_sim = int(validations[0].get("pairs_per_sim", 0))
        if pairs_per_sim <= 0:
            raise SchemaError(f"{study_root} does not report pairs_per_sim")

        studies.append(ResolutionStudy(
            label="E33 material-side" if material_side else "E32 original",
            benchmark=benchmark,
            material_side=material_side,
            seed=next(iter(seeds)),
            checkpoint_epoch=next(iter(epochs)),
            resolutions=expected_resolutions,
            global_rel_l2_pct=tuple(
                float(row["test_rel_l2_norm"]) for row in per_resolution
            ),
            interface_rel_l2_pct=tuple(
                float(row["test_iface_rel_l2_norm"]) for row in per_resolution
            ),
            boundary_rel_l2_pct=tuple(
                float(row["test_boundary_rel_l2_norm"]) for row in per_resolution
            ),
            n_simulations=next(iter(sim_counts)),
            pairs_per_simulation=pairs_per_sim,
            comparison_caveat=str(manifest.get("comparison_caveat", "")),
        ))

    originals = [study for study in studies if not study.material_side]
    variants = [study for study in studies if study.material_side]
    if len(originals) != 1 or len(variants) != 1:
        raise SchemaError(
            "resolution comparison needs one original and one material-side study"
        )
    original, material_side = originals[0], variants[0]
    if original.resolutions != material_side.resolutions:
        raise SchemaError("resolution studies use different evaluation grids")
    if original.n_simulations != material_side.n_simulations:
        raise SchemaError("resolution studies use different simulation counts")
    if original.pairs_per_simulation != material_side.pairs_per_simulation:
        raise SchemaError("resolution studies use different pair counts")
    if original.seed != material_side.seed:
        raise SchemaError("resolution studies use different model seeds")

    drift_payload = _json_payload(drift_paths[0])
    drift_rows = drift_payload.get("per_resolution") or []
    drift_resolutions = tuple(int(row["resolution"]) for row in drift_rows)
    if drift_resolutions != original.resolutions:
        raise SchemaError(
            f"FV drift grids {drift_resolutions} do not match model grids "
            f"{original.resolutions}"
        )
    references = [
        int(row["resolution"]) for row in drift_rows if row.get("is_reference")
    ]
    if len(references) != 1:
        raise SchemaError("FV drift report must identify exactly one reference grid")
    drift = ResolutionDrift(
        resolutions=drift_resolutions,
        global_rel_l2_pct=tuple(float(row["global_rel_l2_pct"])
                                for row in drift_rows),
        interface_rel_l2_pct=tuple(float(row["interface_rel_l2_pct"])
                                   for row in drift_rows),
        boundary_rel_l2_pct=tuple(float(row["boundary_rel_l2_pct"])
                                  for row in drift_rows),
        reference_resolution=references[0],
    )
    return ResolutionStudies(original=original, material_side=material_side,
                             fv_drift=drift)


__all__ = [
    "INVERSE_SENSOR_BENCHMARKS",
    "INVERSE_SENSOR_COUNTS",
    "TEST_RECORD_FIELDS",
    "SCHEMA_V1_COLUMNS",
    "SCHEMA_V2_REQUIRED",
    "SCHEMA_V3_REQUIRED",
    "SCHEMA_V4_REQUIRED",
    "SchemaError",
    "RecordFrame",
    "ResolutionDrift",
    "ResolutionStudies",
    "ResolutionStudy",
    "RolloutArmRecords",
    "artifact_paths",
    "detect_schema_version",
    "seed_from_path",
    "load_test_records",
    "load_test_record_provenance",
    "provenance_path_for",
    "load_many_test_records",
    "load_train_metrics",
    "load_csv_table",
    "load_inverse_sensor_sweep",
    "load_resolution_studies",
    "load_rollout_arms",
    "count_simulations",
    "available_strata",
    "records_by_benchmark",
]
