"""Deterministic, versioned selection of the case a qualitative figure shows.

A truth/prediction/residual panel is a claim about the model, so which case it
shows must be a pre-registered rule rather than an ad hoc choice. This module
replaces the legacy representative-row selector, which ranked *pairs*
(pairs are correlated, so the median pair is not the median simulation), has no
tie-break, and records nothing about what it chose.

Two stages:

1. Rank simulations by their pooled per-simulation metric and take the one at
   ``ceil(quantile * (n - 1))``. Ties break on ascending ``sim_id``, then on a
   PRNG seeded from ``(metric, quantile, strata_filter)``.
2. Pick the snapshot pair within that simulation by an explicit ``pair_rule``.

The result is written to the figure sidecar and printed in the footer. Any
behavioural change requires bumping :data:`PROTOCOL_VERSION`, which a golden
test pins.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

PROTOCOL_VERSION = 1
DEFAULT_TIE_BREAK_SEED = 20260728

PAIR_RULES = ("median_within_sim", "longest_lead", "fixed_leads")


class SelectionError(ValueError):
    """No case satisfies the requested selection."""


@dataclass(frozen=True)
class CaseSelection:
    """The single case a qualitative panel shows, with its full justification."""

    sim_id: int
    s: int
    j: int
    quantile: float
    metric: str
    rank_in_sims: int
    n_sims: int
    sim_metric_value: float
    pair_metric_value: float
    seed: str = ""
    benchmark: str = ""
    pair_rule: str = "median_within_sim"
    strata: dict[str, str] = field(default_factory=dict)
    tie_break_seed: int = DEFAULT_TIE_BREAK_SEED
    protocol_version: int = PROTOCOL_VERSION

    @property
    def label(self) -> str:
        """Footer text, e.g. ``sim 137, rank 22/45 (median of test set)``."""
        name = {0.5: "median", 0.9: "p90", 0.1: "p10"}.get(
            round(self.quantile, 3), f"q{self.quantile:g}")
        return (f"sim {self.sim_id}, rank {self.rank_in_sims}/{self.n_sims} "
                f"({name} of {self.metric})")

    def footer_field(self) -> str:
        """The selection field ``style.add_provenance_footer`` prints."""
        return f"{self.label} [sel v{self.protocol_version}]"

    def to_dict(self) -> dict:
        return asdict(self)


def _strata_key(strata_filter: dict | None) -> str:
    if not strata_filter:
        return ""
    return ";".join(f"{k}={strata_filter[k]}" for k in sorted(strata_filter))


def _tie_break_rng(metric: str, quantile: float, strata_filter: dict | None,
                   seed: int) -> np.random.Generator:
    payload = f"{metric}|{quantile:.6f}|{_strata_key(strata_filter)}|{seed}"
    digest = hashlib.sha256(payload.encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _apply_filter(df: pd.DataFrame, strata_filter: dict | None) -> pd.DataFrame:
    if not strata_filter:
        return df
    mask = pd.Series(True, index=df.index)
    for col, value in strata_filter.items():
        if col not in df.columns:
            raise SelectionError(
                f"strata_filter references missing column {col!r}; available: "
                f"{sorted(df.columns)}"
            )
        mask &= df[col].astype(str) == str(value)
    out = df[mask]
    if out.empty:
        raise SelectionError(f"no rows match strata_filter {strata_filter}")
    return out


def _rank_sims(sim_df: pd.DataFrame, metric: str, quantile: float,
               strata_filter: dict | None, tie_break_seed: int
               ) -> tuple[pd.Series, int, int]:
    values = pd.to_numeric(sim_df[metric], errors="coerce")
    ok = sim_df[values.notna()].copy()
    if ok.empty:
        raise SelectionError(f"no finite {metric} values to rank simulations by")
    ok["_metric"] = pd.to_numeric(ok[metric], errors="coerce")

    rng = _tie_break_rng(metric, quantile, strata_filter, tie_break_seed)
    ok["_jitter"] = rng.random(len(ok))
    ok = ok.sort_values(["_metric", "sim_id", "_jitter"], kind="mergesort")

    n = len(ok)
    index = int(np.ceil(quantile * (n - 1)))
    index = int(np.clip(index, 0, n - 1))
    return ok.iloc[index], index + 1, n


def _select_pair(records: pd.DataFrame, sim_id: int, metric: str,
                 pair_rule: str, strata_filter: dict | None) -> pd.Series:
    part = _apply_filter(records[records["sim_id"] == sim_id], strata_filter)
    if part.empty:
        raise SelectionError(f"simulation {sim_id} has no pairs after filtering")

    if pair_rule == "longest_lead":
        if "t_bar" not in part.columns:
            raise SelectionError("pair_rule 'longest_lead' needs a t_bar column")
        ordered = part.sort_values(["t_bar", "s", "j"], kind="mergesort")
        return ordered.iloc[-1]

    if pair_rule not in ("median_within_sim", "fixed_leads"):
        raise SelectionError(
            f"unknown pair_rule {pair_rule!r}; allowed: {list(PAIR_RULES)}")

    values = pd.to_numeric(part[metric], errors="coerce")
    finite = part[values.notna()].copy()
    if finite.empty:
        raise SelectionError(f"simulation {sim_id} has no finite {metric}")
    finite["_metric"] = pd.to_numeric(finite[metric], errors="coerce")
    target = float(finite["_metric"].median())
    finite["_dist"] = (finite["_metric"] - target).abs()
    ordered = finite.sort_values(["_dist", "s", "j"], kind="mergesort")
    return ordered.iloc[0]


def select_case(sims, records, *, quantile: float = 0.5, metric: str = "rmse_K",
                strata_filter: dict | None = None,
                pair_rule: str = "median_within_sim",
                tie_break_seed: int = DEFAULT_TIE_BREAK_SEED) -> CaseSelection:
    """Pick one ``(sim_id, s, j)`` case at the requested quantile of simulations.

    ``sims`` is a :class:`~visual.pub.stats.SimFrame` (or its frame) and
    ``records`` the pair-level frame it was reduced from. No figure ever
    selects the worst case; ``quantile`` names the case explicitly.
    """
    if not 0.0 <= quantile <= 1.0:
        raise SelectionError(f"quantile must be in [0, 1], got {quantile}")

    sim_df = getattr(sims, "df", sims)
    records = getattr(records, "df", records)
    if metric not in sim_df.columns:
        raise SelectionError(f"{metric!r} is not a per-simulation column")

    sim_df = _apply_filter(sim_df, strata_filter)
    row, rank, n_sims = _rank_sims(sim_df, metric, quantile, strata_filter,
                                   tie_break_seed)
    sim_id = int(row["sim_id"])

    pair_records = records
    if "seed" in records.columns and "seed" in row.index:
        pair_records = records[records["seed"].astype(str) == str(row["seed"])]
    pair = _select_pair(pair_records, sim_id, metric, pair_rule, strata_filter)

    strata = {}
    for col in ("temporal_family", "spatial_family", "regime", "protocol",
                "distribution_class"):
        if col in pair.index and str(pair[col]):
            strata[col] = str(pair[col])
    if strata_filter:
        strata.update({k: str(v) for k, v in strata_filter.items()})

    return CaseSelection(
        sim_id=sim_id,
        s=int(pair["s"]),
        j=int(pair["j"]),
        quantile=float(quantile),
        metric=metric,
        rank_in_sims=rank,
        n_sims=n_sims,
        sim_metric_value=float(row[metric]),
        pair_metric_value=float(pd.to_numeric(pair[metric], errors="coerce")),
        seed=str(row["seed"]) if "seed" in row.index else "",
        benchmark=str(row["benchmark"]) if "benchmark" in row.index else "",
        pair_rule=pair_rule,
        strata=strata,
        tie_break_seed=tie_break_seed,
        protocol_version=PROTOCOL_VERSION,
    )


def select_case_grid(sims, records, *, quantiles=(0.5, 0.9), **kwargs
                     ) -> list[CaseSelection]:
    """One selection per quantile, for figures that show a spread of cases."""
    return [select_case(sims, records, quantile=q, **kwargs) for q in quantiles]


def select_lead_columns(records, selection: CaseSelection, *,
                        lead_quantiles=(0.1, 0.5, 0.9)) -> list[tuple[int, int]]:
    """``(s, j)`` pairs at lead-time quantiles **of the selected simulation itself**.

    The signature figure's early / middle / final columns come from here, so the
    columns are defined relative to that simulation's own ``t_bar`` distribution
    rather than a global lead-time scale.
    """
    records = getattr(records, "df", records)
    part = records[records["sim_id"] == selection.sim_id]
    if selection.seed and "seed" in part.columns:
        part = part[part["seed"].astype(str) == selection.seed]
    if "t_bar" not in part.columns:
        raise SelectionError("select_lead_columns needs a t_bar column")
    part = part[pd.to_numeric(part["t_bar"], errors="coerce").notna()]
    if part.empty:
        raise SelectionError(f"simulation {selection.sim_id} has no finite t_bar")

    t_bar = pd.to_numeric(part["t_bar"], errors="coerce").to_numpy(dtype=np.float64)
    s_col = part["s"].to_numpy()
    j_col = part["j"].to_numpy()

    taken_leads: set[float] = set()
    taken_pairs: set[tuple[int, int]] = set()
    chosen: list[tuple[float, tuple[int, int]]] = []
    for q in lead_quantiles:
        target = float(np.quantile(t_bar, q))
        order = np.lexsort((j_col, s_col, np.abs(t_bar - target)))
        # Distinct lead times, not merely distinct quantiles. When a
        # simulation's leads cluster -- which they do once the pool is narrowed,
        # e.g. to cases with transverse structure -- two quantiles resolve to
        # the same snapshot, and the figure then prints one panel twice under
        # two different labels. Pair distinctness is the fallback for a
        # simulation that genuinely has fewer distinct leads than columns.
        pick = next((i for i in order if float(t_bar[i]) not in taken_leads), None)
        if pick is None:
            pick = next((i for i in order
                         if (int(s_col[i]), int(j_col[i])) not in taken_pairs), None)
        if pick is None:
            raise SelectionError(
                f"simulation {selection.sim_id} has fewer than "
                f"{len(lead_quantiles)} distinct snapshot pairs")
        pair = (int(s_col[pick]), int(j_col[pick]))
        taken_leads.add(float(t_bar[pick]))
        taken_pairs.add(pair)
        chosen.append((float(t_bar[pick]), pair))
    chosen.sort()
    return [pair for _, pair in chosen]


def selection_context(sims, metric: str, selection: CaseSelection) -> np.ndarray:
    """The per-simulation values the selection sits inside, for a context strip."""
    sim_df = getattr(sims, "df", sims)
    if selection.seed and "seed" in sim_df.columns:
        sim_df = sim_df[sim_df["seed"].astype(str) == selection.seed]
    values = pd.to_numeric(sim_df[metric], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


__all__ = [
    "DEFAULT_TIE_BREAK_SEED",
    "PAIR_RULES",
    "PROTOCOL_VERSION",
    "CaseSelection",
    "SelectionError",
    "select_case",
    "select_case_grid",
    "select_lead_columns",
    "selection_context",
]
