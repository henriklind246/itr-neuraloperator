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
    payload = __import__("json").loads(provenance_path.read_text())
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


__all__ = [
    "TEST_RECORD_FIELDS",
    "SCHEMA_V1_COLUMNS",
    "SCHEMA_V2_REQUIRED",
    "SCHEMA_V3_REQUIRED",
    "SCHEMA_V4_REQUIRED",
    "SchemaError",
    "RecordFrame",
    "artifact_paths",
    "detect_schema_version",
    "seed_from_path",
    "load_test_records",
    "load_test_record_provenance",
    "provenance_path_for",
    "load_many_test_records",
    "load_train_metrics",
    "load_csv_table",
    "count_simulations",
    "available_strata",
    "records_by_benchmark",
]
