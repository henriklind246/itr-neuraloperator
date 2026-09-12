"""Inverse-problem figures: F16, F17, F18, F22, F29, F30.

These read inverse result CSVs rather than ``test_records.csv``. F16, F17 and
F22 describe one sensor protocol. F18 consumes the canonical paired sweep from
``scripts/run_inverse_sensor_sweep.py`` and compares the same eight cases at the
three discrete sensor counts 8, 16 and 32.

F29 and F30 read that same sweep but show a *single* pre-registered case, and
descend into its per-sim NPZ artifacts for the profile likelihoods. They are
structural, not distributional: F18 and the inverse tables already carry the
across-case quantities, so these two must not be read as evidence about the
distribution over cases.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visual.pub import panels, records, stats, style
from visual.pub._blocked import blocked

_RECOVERY_KEYS = {
    "forcing": "F16_inverse_forcing_recovery",
    "forcing_itr_sin": "F17_inverse_forcing_itr_sin",
}

# Recovered quantities per inverse benchmark, in the order they are drawn:
# (column stem, axis label). The stem indexes ``<stem>_true`` and the estimate,
# whose suffix differs between the two CSVs -- ``_map`` for the scalar problem,
# ``_hat`` for the profile one.
_FORCING_PARAMS = (("R_c", r"$R_c$"),)

_FORCING_ITR_PARAMS = (
    ("R_base", r"$R_{\mathrm{base}}$"),
    ("A", r"$A$"),
)

_NOMINAL_COVERAGE = 0.95


def _table(source, requirement, key: str, detail: str, required):
    """The single inverse CSV a figure reads, or a blocked figure."""
    paths = records.artifact_paths(source, "inverse_csv")
    if not paths:
        blocked(requirement, detail, key=key)
    df = records.load_csv_table(paths[0], required_columns=tuple(required))
    return df, paths[0]


def _estimate_suffix(df, stem: str) -> str:
    for suffix in ("_map", "_hat"):
        if f"{stem}{suffix}" in df.columns:
            return suffix
    raise KeyError(f"no estimate column for {stem!r} in the inverse table")


def _coverage(df, columns):
    """Interval coverage per method, dropping methods the table does not carry."""
    out = {}
    for label, column in columns.items():
        if column in df.columns:
            out[label] = stats.proportion(df[column])
    return out


def _too_few(n: int) -> list[str]:
    return ["TOO_FEW_SIMS"] if n < stats.MIN_SIMS_FOR_CI else []


def recovery(*, benchmark: str, source=None, spec=None, requirement=None):
    """F16 / F17 -- parameter recovery, one benchmark per figure.

    Each recovered quantity gets a scatter against its true value with the
    identity line, MAP and FV-polished estimates both shown so the reader can
    see how much of the error the surrogate contributes. Nothing is filtered on
    outcome: a case that inverted badly is a point far from the diagonal, not a
    dropped row.

    ``forcing`` recovers a scalar :math:`R_c`, so the figure is one scatter, the
    error distribution, and interval coverage.

    ``forcing_itr_sin`` shows the two resistance parameters separately, with
    one recovery scatter and one absolute-error distribution per parameter.
    """
    key = _RECOVERY_KEYS.get(benchmark, f"recovery[{benchmark}]")
    params = _FORCING_ITR_PARAMS if benchmark == "forcing_itr_sin" else _FORCING_PARAMS
    df, path = _table(
        source, requirement, key,
        f"needs an inverse result CSV for {benchmark}; none resolved from the "
        "manifest.",
        ("sim_id", f"{params[0][0]}_true"))

    suffix = _estimate_suffix(df, params[0][0])
    n_cases = records.count_simulations(df)
    width = spec.width if spec is not None else "two_col"

    n_panels = len(params) + 2
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=style.figsize(width, rows=nrows,
                                                   row_height="std"))
    flat = np.atleast_1d(axes).ravel().tolist()
    color = style.benchmark_color(benchmark)

    for ax, (stem, label) in zip(flat, params):
        polish = df.get(f"{stem}_fvpolish")
        panels.recovery_scatter(
            ax, df[f"{stem}_true"], df[f"{stem}{suffix}"],
            polish=polish, color=color,
            xlabel=f"true {label}", ylabel=f"recovered {label}")
    flat[0].legend(fontsize=5.5, loc="lower right")

    used = len(params)

    if benchmark == "forcing_itr_sin":
        for stem, label in params:
            error = np.abs(np.asarray(df[f"{stem}_true"], dtype=float)
                           - np.asarray(df[f"{stem}{suffix}"], dtype=float))
            panels.ecdf_curves(flat[used], {label: error}, xlabel=f"|error| in {label}",
                               ylabel="fraction of cases", title="parameter recovery error")
            used += 1
    else:
        stem = params[0][0]
        err = np.abs(np.asarray(df[f"{stem}_true"], dtype=float)
                     - np.asarray(df[f"{stem}{suffix}"], dtype=float))
        panels.ecdf_curves(flat[used], {"MAP": err}, xlabel=f"|error| in {label}",
                           ylabel="fraction of cases",
                           title="recovery error")
        used += 1

        rates = _coverage(df, {"MCMC 95%": f"mcmc_{stem}_covered",
                               "profile 95%": f"profile_{stem}_covered"})
        if rates:
            panels.rate_bars(flat[used], rates, ylabel="coverage",
                             title="interval coverage")
            flat[used].axhline(_NOMINAL_COVERAGE, color="0.4", linestyle="--",
                               linewidth=0.8, zorder=1)
            flat[used].set_ylim(0.0, 1.05)
        used += 1

    for ax in flat[used:]:
        ax.set_visible(False)

    style.panel_letters(flat[:used])
    fig.tight_layout()

    metric_definition = {
        "benchmark": benchmark,
        "table": str(path),
        "replication_unit": "inversion case (sim_id)",
        "n_cases": n_cases,
        "estimator": "MAP" if suffix == "_map" else "MAP (profile parameterization)",
        "polish": "FV-polished estimate overlaid where the table carries it",
        "parameters": [p[0] for p in params],
        "failed_cases": "no case is dropped; every inversion in the table is drawn",
        "coverage": "Wilson 95% interval on the fraction of cases whose stated "
                    "95% interval contains the truth"
                    if benchmark != "forcing_itr_sin" else None,
        "identifiability": (
            "squared components of the unit least-determined direction of the "
            "local sensitivity, cases ordered by condition number"
            if benchmark == "forcing_itr_sin" else None),
        "degradations": _too_few(n_cases),
    }
    return fig, None, metric_definition


def sensor_count(*, source=None, spec=None, requirement=None):
    """F18 -- inversion accuracy against sensor count.

    Each row is one benchmark/parameter pair. Columns show absolute error,
    relative error, FV sensor residual, and the 95% parameter profile width.
    Cases remain paired across sensor counts. Width medians and IQRs exclude
    intervals whose crossings were not both found inside the admissible range.
    """
    if not records.artifact_paths(source, "inverse_csv"):
        blocked(
            requirement,
            "needs the paired forcing and forcing_itr_sin sweep at 8, 16, and 32 "
            "sensors; no canonical inverse_sensor_sweep.csv resolved from the "
            "manifest.",
            key="F18_sensor_count",
        )
    table = records.load_inverse_sensor_sweep(source)
    summaries = stats.inverse_sensor_sweep_summaries(table)
    parameter_order = (("forcing", "R_c"), ("forcing_itr_sin", "R_base"), ("forcing_itr_sin", "A"))
    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(
        len(parameter_order), 4,
        figsize=style.figsize(width, rows=len(parameter_order), row_height="grid_row",
                             extra_in=0.35),
    )
    axes = np.atleast_2d(axes)

    for row, (benchmark, parameter) in enumerate(parameter_order):
        summary = summaries[(benchmark, parameter)]
        color = style.benchmark_color(benchmark)
        estimand_label = {"R_c": r"$R_c$", "R_base": r"$R_{\mathrm{base}}$", "A": r"$A$"}[parameter]
        unit_label = "m$^2$ K/W"
        panels.paired_sensor_sweep_panel(
            axes[row, 0], summary.recovery_error, color=color,
            ylabel=f"Absolute error\n[{unit_label}]",
            title=f"{'forcing' if benchmark == 'forcing' else 'forcing ITR'}: {estimand_label}", legend=(row == 0),
        )
        panels.paired_sensor_sweep_panel(
            axes[row, 1], summary.relative_error, color=color,
            ylabel="Relative error [%]", title="parameter recovery", legend=(row == 0),
        )
        panels.paired_sensor_sweep_panel(
            axes[row, 2], summary.fv_residual, color=color,
            ylabel="FV sensor residual [K]", title="physical data fit",
            reference=summary.noise_floor, legend=(row == 0),
        )
        panels.paired_sensor_sweep_panel(
            axes[row, 3], summary.profile_width, color=color,
            ylabel=f"95% profile width\n[{unit_label}]",
            title="profile precision", legend=(row == 0),
        )

    fig.suptitle(
        "Eight paired cases per benchmark; same noise and initialization",
        fontsize=8, y=0.985,
    )
    style.panel_letters(axes.ravel(), loc=(0.0, 1.20))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    paths = records.artifact_paths(source, "inverse_csv")
    metric_definition = {
        "question": (
            "How do the three discrete interface-sensor counts change recovery "
            "accuracy, FV-verified fit, and profile-likelihood precision?"
        ),
        "tables": [str(path) for path in paths],
        "sensor_counts": [8, 16, 32],
        "replication_unit": "inversion case",
        "pairing_fields": ["benchmark", "sim_id", "noise_seed", "init_seed"],
        "pairing": (
            "the identical eight cases, noise realizations, and initialization "
            "seeds are used at every sensor count within each benchmark"
        ),
        "n_cases": {
            benchmark: summaries[(benchmark, parameter)].recovery_error.n_cases
            for benchmark, parameter in parameter_order
        },
        "recovery_estimands": {
            "forcing": "absolute error in scalar R_c",
            "forcing_itr_sin": "separate absolute and relative errors in R_base and A",
        },
        "fv_residual": (
            "physical RMS sensor residual in Kelvin from the real finite-volume "
            "solver evaluated at the recovered parameter"
        ),
        "profile_interval": (
            "width of each 95% parameter profile interval, endpoints bracketed and bisected with the nuisance re-optimized at every pinned value; bound-limited intervals are marked and excluded from width summaries"
        ),
        "bound_limited_cases_by_sensor_count": {
            f"{benchmark}:{parameter}": {
                str(count): int(value)
                for count, value in zip(
                    summaries[(benchmark, parameter)].profile_width.sensor_counts,
                    summaries[(benchmark, parameter)].bound_limited_cases,
                )
            }
            for benchmark, parameter in parameter_order
        },
        "summary": (
            "paired case trajectories plus median and IQR across cases at each "
            "discrete count; connecting segments are not fitted curves"
        ),
        "inferential_interval": (
            "none for the across-case median: the per-case parameter intervals use profile likelihood"
        ),
        "degradations": ["TOO_FEW_SIMS_NO_CI"],
    }
    return fig, None, metric_definition


_ITR_SIN = "forcing_itr_sin"
_RC_PROFILE_SAMPLES = 201

# Dependent ceiling of the sinusoidal parameterization, A <= R_PEAK_MAX - R_base.
_RC_PEAK_MAX = 3.0

_ITR_SIN_LABELS = {"R_base": r"$R_{\mathrm{base}}$", "A": r"$A$"}
_RC_UNIT = r"m$^2$ K/W"


def _itr_sin_sweep(source, requirement, key: str):
    """The forcing_itr_sin sweep table, its CSV path, and the case to show."""
    paths = records.artifact_paths(source, "inverse_csv", benchmark=_ITR_SIN)
    if not paths:
        blocked(
            requirement,
            "needs the paired forcing_itr_sin sweep at 8, 16, and 32 sensors; "
            "no inverse sweep CSV for that benchmark resolved from the "
            "manifest.",
            key=key,
        )
    # Restricted to the one arm these figures read. F18 loads both and needs
    # both; failing here because the scalar-R_c sweep is absent would report a
    # reason that has nothing to do with whether this figure can be drawn.
    table = records.load_inverse_sensor_sweep(source, benchmarks=(_ITR_SIN,))
    sim_id = stats.representative_inverse_case(
        table, benchmark=_ITR_SIN, reference_sensors=8)
    return table, paths[0], int(sim_id)


def _case_row(table, sim_id: int, n_sensors: int):
    rows = table.loc[
        (table["benchmark"].astype(str) == _ITR_SIN)
        & (table["sim_id"].astype(int) == int(sim_id))
        & (table["n_sensors"].astype(int) == int(n_sensors))
    ]
    if len(rows) != 1:
        raise records.SchemaError(
            f"{_ITR_SIN} sim {sim_id} has {len(rows)} rows at {n_sensors} "
            "sensors; the paired sweep must carry exactly one"
        )
    return rows.iloc[0]


def rc_profile_recovery(*, source=None, spec=None, requirement=None):
    """F29 -- recovered :math:`R_c(y)` against the truth as sensors are added.

    One inversion case at 8, 16 and 32 interface sensors, two curves per panel,
    identical axes throughout, so the only thing that changes left to right is
    the sensor count. The case is picked by the pre-registered rule in
    ``stats.representative_inverse_case`` -- the lower-median case by recovery
    error at the hardest arm -- rather than by looking at the drawn result.

    No uncertainty band is drawn around the recovered profile. The two 95%
    parameter intervals are marginal and correlated, so propagating them
    independently onto :math:`R_c(y)` would assert a band the inversion never
    computed. F30 panel (c) carries the joint statement instead.

    Aggregate accuracy over all eight cases belongs to F18 and the inverse
    tables; this figure exists to make those numbers physically legible.
    """
    from src.physics.internal_source import make_rc_sin_profile

    key = "F29_inverse_rc_profile_recovery"
    table, path, sim_id = _itr_sin_sweep(source, requirement, key)

    counts = records.INVERSE_SENSOR_COUNTS
    rows = [_case_row(table, sim_id, count) for count in counts]

    truths = np.array([[float(r["R_base_true"]), float(r["A_true"])]
                       for r in rows])
    if not np.allclose(truths, truths[0]):
        raise records.SchemaError(
            f"{_ITR_SIN} sim {sim_id} carries different true (R_base, A) at "
            "different sensor counts; the sweep arms are not paired"
        )

    y = np.linspace(0.0, 1.0, _RC_PROFILE_SAMPLES)
    truth = make_rc_sin_profile(y, truths[0, 0], truths[0, 1])
    estimates = [
        make_rc_sin_profile(y, float(row["R_base_hat"]), float(row["A_hat"]))
        for row in rows
    ]

    # One R_c range for all three panels: rescaling per panel would give a
    # worse reconstruction the same visual gap as a better one.
    stack = np.concatenate([truth, *estimates])
    low, high = float(stack.min()), float(stack.max())
    pad = 0.08 * (high - low) or 0.05
    ylim = (low - pad, high + pad)

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, len(counts),
                             figsize=style.figsize(width, row_height="std"))
    flat = np.atleast_1d(axes).ravel().tolist()
    color = style.benchmark_color(_ITR_SIN)

    for index, (ax, count, rc_hat) in enumerate(zip(flat, counts, estimates)):
        panels.rc_profile_recovery_panel(
            ax, y, truth, rc_hat, color=color,
            title=f"{count} interface sensors",
            ylabel=f"$R_c(y)$ [{_RC_UNIT}]" if index == 0 else "",
            legend=(index == 0))
        ax.set_ylim(*ylim)

    fig.suptitle(f"One inversion case (sim {sim_id}) at three sensor counts",
                 fontsize=8, y=0.99)
    style.panel_letters(flat)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    metric_definition = {
        "question": (
            "As interface sensors are added, does the inversion recover the "
            "shape of R_c(y) or only its scale?"
        ),
        "benchmark": _ITR_SIN,
        "table": str(path),
        "replication_unit": "one inversion case (sim_id)",
        "n_cases": 1,
        "sim_id": sim_id,
        "sensor_counts": list(counts),
        "selection_rule": (
            "cases ranked by absolute error in the last recovered parameter "
            "at the 8-sensor arm; the lower median is shown. Fixed in "
            "stats.representative_inverse_case and applied without consulting "
            "the drawn result."
        ),
        "estimator": (
            "MAP in the unconstrained parameterization, FNO surrogate forward "
            "model"
        ),
        "forward_model": "R_c(y) = R_base + A sin(pi y)",
        "true_curve": {"R_base": float(truths[0, 0]), "A": float(truths[0, 1])},
        "recovered_curves": {
            int(count): {"R_base": float(row["R_base_hat"]),
                         "A": float(row["A_hat"])}
            for count, row in zip(counts, rows)
        },
        "axes": "identical y and R_c limits on all three panels",
        "no_uncertainty_band": (
            "the two 95% profile intervals are marginal and correlated, so no "
            "band is propagated from them onto R_c(y); the joint statement is "
            "F30 panel (c)"
        ),
        "aggregate_evidence": (
            "across-case accuracy is reported by F18 and the inverse tables; "
            "this figure is a single-case structural view and is not evidence "
            "about the distribution over cases"
        ),
        "inferential_interval": (
            "none; a single case carries no across-case interval"
        ),
        "degradations": ["SINGLE_CASE_ILLUSTRATION"],
    }
    selection = {
        "sim_id": sim_id,
        "rule": "lower-median absolute recovery error at the 8-sensor arm",
    }
    return fig, selection, metric_definition


def identifiability(*, n_sensors: int = 8, source=None, spec=None,
                    requirement=None):
    """F30 -- are :math:`R_{\\mathrm{base}}` and :math:`A` separately identifiable?

    Panels (a) and (b) are the two 95% profile likelihoods for the same case
    F29 draws, each with its true value, its recovered value, and the
    chi-squared threshold whose crossings define the reported interval. Panel
    (c) contours the joint region over the same pair: a long tilted region says
    the two parameters trade against each other, so a small marginal error can
    coexist with a poorly determined profile; a compact one says they are
    separately determined.

    The joint region uses the 2-parameter chi-squared threshold. Taking the
    product of the two 1-D intervals instead would understate the region
    exactly where the parameters are correlated -- which is the case this
    figure exists to detect.

    Blank cells are inadmissible: the parameterization enforces
    ``A <= R_PEAK_MAX - R_base``, so those pairs were never evaluated.
    """
    key = "F30_inverse_identifiability"
    if int(n_sensors) not in records.INVERSE_SENSOR_COUNTS:
        raise ValueError(
            f"n_sensors={n_sensors} is not one of the swept counts "
            f"{records.INVERSE_SENSOR_COUNTS}"
        )
    table, path, sim_id = _itr_sin_sweep(source, requirement, key)
    del table

    artifact_path = records.inverse_artifact_path(
        path, n_sensors=int(n_sensors), sim_id=sim_id)
    try:
        artifact = records.load_inverse_profile_artifact(artifact_path)
    except records.SchemaError as exc:
        blocked(
            requirement,
            "needs the per-sim profile-likelihood artifact for the "
            f"representative case: {exc}",
            key=key,
        )

    degradations = ["SINGLE_CASE_ILLUSTRATION"]
    region = artifact.joint
    if region is None:
        if artifact.observation_jacobian is None or artifact.sigma_eff2 is None:
            blocked(
                requirement,
                f"{artifact_path} carries neither a measured joint NLL grid "
                "nor the sensitivity needed for the quadratic fallback, so "
                "the joint region cannot be drawn.",
                key=key,
            )
        region = stats.laplace_joint_region(
            artifact.observation_jacobian, artifact.sigma_eff2,
            artifact.theta_hat, param_names=("R_base", "A"),
            level=artifact.level, bounds=artifact.theta_bounds,
            peak_max=_RC_PEAK_MAX)
        degradations.append("JOINT_REGION_LAPLACE_APPROXIMATION")

    order = [artifact.param_names.index(name) for name in region.param_names]
    theta_hat = np.asarray(artifact.theta_hat, dtype=float)[order]
    theta_true = np.asarray(artifact.theta_true, dtype=float)[order]

    width = spec.width if spec is not None else "two_col"
    fig, axes = plt.subplots(1, 3,
                             figsize=style.figsize(width, row_height="std"))
    flat = np.atleast_1d(axes).ravel().tolist()
    color = style.benchmark_color(_ITR_SIN)

    bound_limited = []
    for index, name in enumerate(("R_base", "A")):
        curve = artifact.profile(name)
        if curve.bound_limited:
            bound_limited.append(name)
        panels.profile_likelihood_panel(
            flat[index], curve, color=color,
            xlabel=f"{_ITR_SIN_LABELS[name]} [{_RC_UNIT}]",
            ylabel=r"$\Delta\ell_p$" if index == 0 else "",
            title=f"profile likelihood, {_ITR_SIN_LABELS[name]}",
            threshold=artifact.profile_threshold,
            legend=(index == 0))

    filled = panels.joint_nll_contour_panel(
        flat[2], region, theta_hat=theta_hat, theta_true=theta_true,
        labels=tuple(f"{_ITR_SIN_LABELS[name]} [{_RC_UNIT}]"
                     for name in region.param_names),
        title="joint region"
              + (" (quadratic approx.)" if region.approximate else ""),
        legend=True)
    panels.attach_colorbar(fig, filled, flat[2], r"$\Delta\ell$")

    fig.suptitle(
        f"One inversion case (sim {sim_id}) at {int(n_sensors)} sensors",
        fontsize=8, y=0.99)
    style.panel_letters(flat)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    metric_definition = {
        "question": (
            "Are R_base and A separately identifiable from the interface "
            "sensor data, or do they compensate for each other?"
        ),
        "benchmark": _ITR_SIN,
        "table": str(path),
        "artifact": str(artifact_path),
        "replication_unit": "one inversion case (sim_id)",
        "n_cases": 1,
        "sim_id": sim_id,
        "n_sensors": int(n_sensors),
        "selection_rule": (
            "the same case F29 draws: cases ranked by absolute error in the "
            "last recovered parameter at the 8-sensor arm, lower median taken. "
            "Fixed in stats.representative_inverse_case."
        ),
        "estimator": (
            f"profile likelihood at level {artifact.level}, sum-convention "
            "NLL 0.5 * sum(resid^2) / sigma_eff2"
        ),
        "marginal_threshold": (
            f"Delta-ell = chi2.ppf({artifact.level}, 1) / 2 = "
            f"{artifact.profile_threshold:.4f}; interval endpoints are its "
            "crossings, with the nuisance parameter re-optimized at every "
            "pinned value"
        ),
        "joint_threshold": (
            f"Delta-ell = chi2.ppf({artifact.level}, 2) / 2 = "
            f"{float(region.threshold):.4f}; the 2-parameter threshold, not "
            "the product of the two marginal intervals, which would understate "
            "the region wherever the parameters are correlated"
        ),
        "joint_region_source": (
            "Gauss-Newton quadratic approximation H = J^T J / sigma_eff2; "
            "curvature beyond second order is not shown"
            if region.approximate else
            "measured NLL grid with both parameters pinned"
        ),
        "joint_condition_number": float(region.cond_number),
        "inadmissible_region": (
            "blank cells are pairs the parameterization forbids "
            f"(A <= {_RC_PEAK_MAX} - R_base); they were not evaluated"
        ),
        "bound_limited_parameters": bound_limited,
        "inferential_interval": (
            "profile-likelihood intervals for this one case; no across-case "
            "interval is claimed"
        ),
        "degradations": degradations,
    }
    selection = {"sim_id": sim_id, "n_sensors": int(n_sensors)}
    return fig, selection, metric_definition


def surrogate_fidelity(*, source=None, spec=None, requirement=None):
    """F22 -- surrogate fidelity against recovery error (supplemental).

    Two panels, both against the scalar recovery error: the forward FNO-FV
    mismatch at the recovered parameter, and the data-fit residual the
    optimizer actually minimized. The question is whether inversion failures
    are explained by surrogate error or by genuine ill-posedness. A weak rank
    correlation says a better surrogate would not fix them, which makes the
    identifiability panel of F17 the real story.

    Rank rather than linear correlation, and no fitted line: with ten cases a
    regression line would assert more structure than the data carries.
    """
    df, path = _table(
        source, requirement, "F22_surrogate_fidelity",
        "needs an inverse result CSV carrying per-case forward mismatch "
        "alongside recovery error.",
        ("sim_id", "R_c_abs_error"))

    series = [c for c in ("fno_vs_fv_resid", "fno_resid") if c in df.columns]
    if not series:
        blocked(requirement,
                "the inverse table carries no forward-mismatch column "
                "(fno_vs_fv_resid, fno_resid), so there is nothing to correlate "
                "recovery error against.",
                key="F22_surrogate_fidelity")

    titles = {
        "fno_vs_fv_resid": "surrogate mismatch",
        "fno_resid": "data-fit residual",
    }
    labels = {
        "fno_vs_fv_resid": r"$\|u_{\mathrm{FNO}} - u_{\mathrm{FV}}\|$ at MAP",
        "fno_resid": "FNO data residual at MAP",
    }

    width = spec.width if spec is not None else "one_half_col"
    fig, axes = plt.subplots(1, len(series),
                             figsize=style.figsize(width, row_height="std"))
    flat = np.atleast_1d(axes).ravel().tolist()

    err = df["R_c_abs_error"]
    correlations = {}
    for ax, column in zip(flat, series):
        rho = stats.spearman(df[column], err)
        correlations[column] = rho
        panels.association_scatter(
            ax, df[column], err, correlation=rho,
            color=style.benchmark_color("forcing"),
            xlabel=labels[column], ylabel=r"$|R_c$ recovery error$|$",
            title=titles[column])

    style.panel_letters(flat)
    fig.tight_layout()

    n_cases = records.count_simulations(df)
    metric_definition = {
        "table": str(path),
        "replication_unit": "inversion case (sim_id)",
        "n_cases": n_cases,
        "association": "Spearman rank correlation, percentile bootstrap over "
                       "cases",
        "no_fit_line": "monotone association only; no functional form is "
                       "asserted at this case count",
        "rho": {c: float(r.rho) for c, r in correlations.items()},
        "degradations": sorted(
            set(_too_few(n_cases))
            | {d.code for r in correlations.values() for d in r.degradations}),
    }
    return fig, None, metric_definition


__all__ = [
    "identifiability",
    "rc_profile_recovery",
    "recovery",
    "sensor_count",
    "surrogate_fidelity",
]
