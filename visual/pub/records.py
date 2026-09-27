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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.operators.record_schema import TEST_RECORD_FIELDS
from visual.pub.manifest import Degradation

# The 26 columns actually present in the surviving v1 artifact
# (runs/source_itr_sin_smoke3/config0/seed42/test_records.csv).
SCHEMA_V1_COLUMNS: tuple[str, ...] = (
    "sim_id", "s", "j", "t_s", "t_bar", "R_c", "benchmark",
    "temporal_family", "spatial_family",
    "x_h", "y_h", "A", "freq", "regime",
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
    """Field and node-jump GNRMSE of one fixed checkpoint per evaluation grid."""

    benchmark: str
    material_side: bool
    seed: str
    checkpoint_epoch: int
    resolutions: tuple[int, ...]
    field_gnrmse_pct: tuple[float, ...]
    node_jump_gnrmse_pct: tuple[float, ...]
    n_simulations: int
    pairs_per_simulation: int
    checkpoint_sha256: str


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


def load_global_field_records(source):
    """Load F27/F28 records and evaluation metadata without loading a model."""
    import numpy as np
    import yaml
    from visual.pub.manifest import _inspect_artifact, sha256_file

    by_benchmark, metadata, extra_sources = {}, {}, []
    runs = {Path(run.run_dir).resolve(): run for run in getattr(source, "runs", ())}
    for ref in list(getattr(source, "artifacts", ())):
        if ref.kind != "test_records":
            continue
        path = Path(ref.path)
        record = load_test_records(path, seed=ref.seed)
        if record.schema_version < 2:
            raise SchemaError(f"{path}: F27/F28 require schema-v2 pooled statistics")
        if len(record.benchmarks) != 1:
            raise SchemaError(f"{path}: expected one benchmark per test-record artifact")
        benchmark = record.benchmarks[0]
        seed = str(record.seed)
        if set(record.df["seed"]) != {seed}:
            raise SchemaError(f"{path}: record seed does not match the manifest seed")
        if seed in metadata.get(benchmark, {}):
            raise SchemaError(f"{benchmark}: duplicate record files for seed {seed}")
        run = runs.get(path.parent.resolve())
        if run is not None and run.benchmark != benchmark:
            raise SchemaError(f"{path}: benchmark disagrees with config_used.yaml")
        config_path = path.parent / "config_used.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        meta = {"representation": run.representation if run else None}
        provenance = None
        if record.schema_version >= 4:
            provenance = load_test_record_provenance(path)
            if provenance.get("benchmark") != benchmark or str(provenance.get("seed")) != seed:
                raise SchemaError(f"{path}: provenance benchmark/seed mismatch")
            if provenance.get("prediction_mode") != "direct_pair" or provenance.get("rollout_enabled"):
                raise SchemaError(f"{path}: F27/F28 require direct prediction records")
            if run is not None and run.representation != provenance.get("representation"):
                raise SchemaError(f"{path}: representation disagrees with run provenance")
            population = provenance["evaluation_population"]
            meta.update({"t_grid": population["t_grid"],
                         "time_grid_source": str(provenance_path_for(path)),
                         "prediction_mode": provenance["prediction_mode"],
                         "evaluation_population_hash": provenance["evaluation_population_hash"],
                         "representation": provenance.get("representation")})
            extra_sources.append(_inspect_artifact(provenance_path_for(path), "json_report", seed=seed))
        elif (config.get("evaluation", {}).get("rollout") or {}).get("enabled"):
            raise SchemaError(f"{path}: legacy run config enables rollout; direct records required")
        if config_path.exists():
            extra_sources.append(_inspect_artifact(config_path, "json_report", seed=seed))
        data = config.get("data", {})
        for axis in ("t", "y"):
            raw = data.get(f"{axis}_grid_path")
            if not raw:
                continue
            grid_path = Path(raw)
            if not grid_path.is_absolute():
                from visual.pub.manifest import PROJECT_ROOT
                grid_path = PROJECT_ROOT / grid_path
            if not grid_path.exists():
                continue
            if provenance is not None:
                expected = provenance["evaluation_population"]["dataset_file_hashes"].get(f"{axis}_grid")
                if expected != sha256_file(grid_path):
                    raise SchemaError(f"{grid_path}: grid differs from the evaluated dataset")
            grid = np.load(grid_path, allow_pickle=False)
            if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
                raise SchemaError(f"{grid_path}: invalid coordinate grid")
            if axis == "t":
                if "t_grid" in meta and not np.array_equal(grid, meta["t_grid"]):
                    raise SchemaError(f"{grid_path}: time grid disagrees with provenance")
                meta.update(t_grid=grid.tolist(), time_grid_source=str(grid_path))
            else:
                meta.update(y_bounds=[float(grid[0]), float(grid[-1])], bounds_source=str(grid_path))
            extra_sources.append(_inspect_artifact(grid_path, "json_report", seed=seed))
        by_benchmark.setdefault(benchmark, []).append(record.df)
        metadata.setdefault(benchmark, {})[seed] = meta
    known = {ref.path for ref in getattr(source, "artifacts", ())}
    for ref in extra_sources:
        if ref.path not in known:
            source.artifacts.append(ref)
            known.add(ref.path)
    return {b: pd.concat(parts, ignore_index=True) for b, parts in by_benchmark.items()}, metadata


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
INVERSE_SENSOR_BENCHMARKS = ("forcing", "forcing_itr_sin")
_INVERSE_SENSOR_BASE_COLUMNS = (
    "benchmark", "sim_id", "n_sensors", "noise_seed", "init_seed",
    "noise_std_K", "fv_resid_rms_K", "fv_resid_over_noise",
)
_INVERSE_SENSOR_COLUMNS = {
    benchmark: tuple(column for name in names for column in (
        f"{name}_true", f"{name}_map" if name == "R_c" else f"{name}_hat",
        f"{name}_abs_error" if name == "R_c" else f"{name}_abserr",
        f"{name}_rel_error_pct", f"profile_{name}_ci_low", f"profile_{name}_ci_high",
        f"profile_{name}_bound_limited",
    ))
    for benchmark, names in {"forcing": ("R_c",), "forcing_itr_sin": ("R_base", "A")}.items()
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

    unexpected = sorted(set(table["benchmark"]) - set(INVERSE_SENSOR_BENCHMARKS))
    if unexpected:
        raise SchemaError(
            f"inverse sensor sweep contains unsupported benchmarks {unexpected}"
        )
    # ``benchmarks`` narrows what is returned, so a single-arm consumer (F29,
    # F30) works against either layout: its own CSV, or the combined table that
    # also carries the arm it does not read.
    table = table.loc[table["benchmark"].isin(benchmarks)].reset_index(drop=True)

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
            if column.endswith("_bound_limited"):
                values = part[column].map({True: True, False: False, 1: True, 0: False,
                    "True": True, "False": False, "true": True, "false": False,
                    "1": True, "0": False, "1.0": True, "0.0": False})
            else:
                values = pd.to_numeric(part[column], errors="coerce")
            if values.isna().any() and not column.endswith("_rel_error_pct"):
                raise SchemaError(f"inverse sensor sweep for {benchmark} has invalid {column}")
            table.loc[part.index, column] = values

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


# The per-sim inverse artifacts written by scripts/invert.py --artifact-dir.
# Schema 3 carries the per-parameter profile likelihoods; 4 adds the optional
# joint_nll_* block (--joint-nll-grid). Readers that only need the profiles
# accept >= 3 and treat the joint block as absent.
INVERSE_ARTIFACT_MIN_VERSION = 3

# A stored profile value below its own reference means the reported optimum was
# not the optimum, which invalidates the interval rather than merely perturbing
# it. Matches scripts/invert.PROFILE_OPTIMUM_ATOL.
_PROFILE_OPTIMUM_ATOL = 1e-5


@dataclass(frozen=True)
class ProfileCurve:
    """One parameter's profile likelihood, as stored by ``scripts/invert.py``.

    ``delta_ell`` is ``nll - nll_min`` on the sum convention
    ``0.5 * sum(resid^2) / sigma_eff2``, so it is directly comparable against
    ``chi2.ppf(level, 1) / 2``.
    """

    param_name: str
    param_index: int
    values: np.ndarray
    delta_ell: np.ndarray
    theta_hat: float
    theta_true: float
    ci_low: float
    ci_high: float
    bound_limited: bool


@dataclass(frozen=True)
class JointNLLGrid:
    """The measured joint ``delta_ell`` surface over a 2-parameter space.

    Rows index ``param_names[0]``, columns index ``param_names[1]``. Entries are
    ``nan`` where the pair violates the dependent ceiling, so the admissible set
    is triangular and must be left blank rather than extrapolated.
    ``approximate`` is always ``False`` here; the Gauss-Newton fallback in
    ``visual.pub.stats`` produces the same duck type with ``True``.
    """

    param_names: tuple[str, str]
    axes: tuple[np.ndarray, np.ndarray]
    delta_ell: np.ndarray
    threshold: float
    approximate: bool = False
    cond_number: float = float("nan")


@dataclass(frozen=True)
class InverseProfileArtifact:
    """One ``sim_{id:05d}.npz`` from an inverse run, validated and decoded."""

    path: Path
    schema_version: int
    benchmark: str
    sim_id: int
    param_names: tuple[str, ...]
    theta_hat: np.ndarray
    theta_true: np.ndarray
    theta_bounds: np.ndarray
    param_scales: np.ndarray
    level: float
    profile_threshold: float
    profiles: tuple[ProfileCurve, ...]
    observation_jacobian: np.ndarray | None
    sigma_eff2: float | None
    joint: JointNLLGrid | None

    def profile(self, param_name: str) -> ProfileCurve:
        for curve in self.profiles:
            if curve.param_name == param_name:
                return curve
        raise SchemaError(
            f"{self.path} has no profile for {param_name!r}; "
            f"stored profiles are {[c.param_name for c in self.profiles]}"
        )


def inverse_artifact_path(csv_path: str | Path, *, n_sensors: int,
                          sim_id: int) -> Path:
    """Per-sim NPZ path for one arm of an inverse sensor sweep.

    The layout is fixed by ``scripts/run_inverse_sensor_sweep.py`` and enforced
    by its ``_validate_artifacts``, so it is derived from the resolved combined
    CSV rather than declared 24 more times in the manifest. Keeping the
    derivation here means exactly one place breaks if the sweep layout changes.
    """
    return (
        Path(csv_path).parent
        / f"sensors_{int(n_sensors):02d}"
        / "artifacts"
        / f"sim_{int(sim_id):05d}.npz"
    )


def _scalar(stored, key: str) -> float:
    return float(np.asarray(stored[key]).reshape(()))


def _flag(stored, key: str) -> bool:
    return bool(np.asarray(stored[key]).reshape(()))


def load_inverse_profile_artifact(
    path: str | Path, *, require_version: int = INVERSE_ARTIFACT_MIN_VERSION
) -> InverseProfileArtifact:
    """Load one per-sim inverse artifact, including its profile likelihoods.

    Raises :class:`SchemaError` for a missing file, a version below the floor, or
    a profile block that is absent or internally inconsistent. A figure that
    cannot show the real likelihood must be blocked, not quietly approximated.
    """
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"inverse artifact not found: {path}")

    with np.load(path, allow_pickle=False) as stored:
        keys = set(stored.files)
        if "artifact_schema_version" not in keys:
            raise SchemaError(f"{path} carries no artifact_schema_version")
        version = int(_scalar(stored, "artifact_schema_version"))
        if version < require_version:
            raise SchemaError(
                f"{path} is artifact schema v{version}; "
                f"v{require_version} or later is required"
            )

        param_names = tuple(
            str(name) for name in np.asarray(stored["param_names"]).ravel()
        )
        theta_hat = np.asarray(stored["theta_hat"], dtype=np.float64).ravel()
        theta_true = np.asarray(stored["theta_true"], dtype=np.float64).ravel()
        if len(param_names) != theta_hat.size or theta_hat.size != theta_true.size:
            raise SchemaError(
                f"{path} has {len(param_names)} parameter names but "
                f"theta_hat/theta_true of size {theta_hat.size}/{theta_true.size}"
            )
        if "profile_threshold" not in keys or "profile_level" not in keys:
            raise SchemaError(f"{path} stores no profile likelihood block")
        threshold = _scalar(stored, "profile_threshold")
        level = _scalar(stored, "profile_level")

        curves = []
        for index, name in enumerate(param_names):
            stem = f"profile_{name}"
            missing = [
                f"{stem}_{suffix}" for suffix in
                ("grid", "nll", "nll_min", "ci_low", "ci_high", "bound_limited")
                if f"{stem}_{suffix}" not in keys
            ]
            if missing:
                raise SchemaError(f"{path} is missing profile keys {missing}")
            values = np.asarray(stored[f"{stem}_grid"], dtype=np.float64)
            nll = np.asarray(stored[f"{stem}_nll"], dtype=np.float64)
            if values.shape != nll.shape or values.size < 2:
                raise SchemaError(
                    f"{path} profile for {name} has grid/nll shapes "
                    f"{values.shape}/{nll.shape}"
                )
            delta = nll - _scalar(stored, f"{stem}_nll_min")
            if float(delta.min()) < -_PROFILE_OPTIMUM_ATOL:
                raise SchemaError(
                    f"{path} profile for {name} dips {-float(delta.min()):.3g} "
                    "below its own reference: the stored fit is not the optimum "
                    "and the interval is not usable"
                )
            curves.append(ProfileCurve(
                param_name=name, param_index=index, values=values,
                delta_ell=np.clip(delta, 0.0, None),
                theta_hat=float(theta_hat[index]),
                theta_true=float(theta_true[index]),
                ci_low=_scalar(stored, f"{stem}_ci_low"),
                ci_high=_scalar(stored, f"{stem}_ci_high"),
                bound_limited=_flag(stored, f"{stem}_bound_limited"),
            ))

        joint = None
        joint_axis_keys = tuple(f"joint_nll_grid_{name}" for name in param_names)
        if "joint_nll" in keys:
            if len(param_names) != 2 or not set(joint_axis_keys) <= keys:
                raise SchemaError(
                    f"{path} stores joint_nll without the matching "
                    f"{list(joint_axis_keys)} axes"
                )
            axes = tuple(
                np.asarray(stored[key], dtype=np.float64) for key in joint_axis_keys
            )
            surface = np.asarray(stored["joint_nll"], dtype=np.float64)
            if surface.shape != (axes[0].size, axes[1].size):
                raise SchemaError(
                    f"{path} joint_nll has shape {surface.shape}; expected "
                    f"{(axes[0].size, axes[1].size)} from its axes"
                )
            joint_min = _scalar(stored, "joint_nll_min")
            delta = surface - joint_min
            if np.nanmin(delta) < -_PROFILE_OPTIMUM_ATOL:
                raise SchemaError(
                    f"{path} joint_nll dips below its own reference; the stored "
                    "optimum is wrong and the region would be misplaced"
                )
            joint = JointNLLGrid(
                param_names=(param_names[0], param_names[1]),
                axes=axes,
                # nan marks the inadmissible corner and must survive the clip so
                # the panel leaves it blank instead of filling it at zero.
                delta_ell=np.where(np.isnan(delta), np.nan, np.clip(delta, 0.0, None)),
                threshold=_scalar(stored, "joint_threshold"),
            )

        return InverseProfileArtifact(
            path=path,
            schema_version=version,
            benchmark=str(np.asarray(stored["benchmark"]).reshape(())),
            sim_id=int(_scalar(stored, "sim_id")),
            param_names=param_names,
            theta_hat=theta_hat,
            theta_true=theta_true,
            theta_bounds=np.asarray(stored["theta_bounds"], dtype=np.float64),
            param_scales=np.asarray(stored["param_scales"], dtype=np.float64),
            level=level,
            profile_threshold=threshold,
            profiles=tuple(curves),
            observation_jacobian=(
                np.asarray(stored["observation_jacobian"], dtype=np.float64)
                if "observation_jacobian" in keys else None
            ),
            sigma_eff2=(
                _scalar(stored, "sigma_eff2") if "sigma_eff2" in keys else None
            ),
            joint=joint,
        )


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


RESOLUTION_STUDY_FILENAME = "resolution_study.json"
RESOLUTION_STUDY_METRICS = ("field_gnrmse_pct", "node_jump_gnrmse_pct")


def _json_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"JSON report must contain an object: {path}")
    return payload


def load_resolution_study(source) -> ResolutionStudy:
    """Load the one ``scripts/run_resolution_study.py`` report of a source."""
    paths = [
        p for p in artifact_paths(source, "json_report")
        if p.name == RESOLUTION_STUDY_FILENAME
    ]
    if len(paths) != 1:
        raise SchemaError(
            f"resolution source needs one {RESOLUTION_STUDY_FILENAME}, "
            f"found {len(paths)}"
        )
    payload = _json_payload(paths[0])
    rows = payload.get("per_resolution") or []
    if not rows:
        raise SchemaError(f"{paths[0]} reports no resolutions")
    resolutions = tuple(int(row["resolution"]) for row in rows)
    if list(resolutions) != sorted(set(resolutions)):
        raise SchemaError(
            f"{paths[0]} resolutions must be unique and ascending: {resolutions}"
        )
    values: dict[str, tuple[float, ...]] = {}
    for metric in RESOLUTION_STUDY_METRICS:
        try:
            values[metric] = tuple(float(row[metric]) for row in rows)
        except KeyError as exc:
            raise SchemaError(f"{paths[0]} is missing {metric}") from exc
        if not all(np.isfinite(values[metric])):
            raise SchemaError(f"{paths[0]} has non-finite {metric}")

    pairs_per_sim = int(payload.get("pairs_per_sim", 0))
    n_simulations = int(payload.get("num_sims", 0))
    if pairs_per_sim <= 0 or n_simulations <= 0:
        raise SchemaError(f"{paths[0]} does not report its simulation/pair counts")
    return ResolutionStudy(
        benchmark=str(payload.get("benchmark", "")),
        material_side=bool(payload.get("material_side", False)),
        seed=str(payload.get("checkpoint_seed", "")),
        checkpoint_epoch=int(payload["checkpoint_epoch"]),
        resolutions=resolutions,
        field_gnrmse_pct=values["field_gnrmse_pct"],
        node_jump_gnrmse_pct=values["node_jump_gnrmse_pct"],
        n_simulations=n_simulations,
        pairs_per_simulation=pairs_per_sim,
        checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
    )


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
    "RESOLUTION_STUDY_FILENAME",
    "RESOLUTION_STUDY_METRICS",
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
    "load_resolution_study",
    "load_rollout_arms",
    "count_simulations",
    "available_strata",
    "records_by_benchmark",
]
