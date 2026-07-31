"""Training-curve figure: F03.

Blocked on multi-seed ``train_metrics.csv`` for all four benchmarks. This is the
one tier-1 figure that does not need ``test_records.csv``, so it unblocks as
soon as the canonical training runs finish, before eval is re-run.
"""

from __future__ import annotations

from visual.pub._blocked import blocked


def learning_curves(*, source=None, spec=None, requirement=None):
    """F03 -- learning curves across the four benchmarks.

    Design, once the artifacts exist:

    * One panel per benchmark, train and validation ``rel_l2`` against epoch,
      one line per model seed plus a median line across seeds.
    * Best-epoch marker per seed, taken from the run's ``final_metrics.json``
      rather than recomputed, so the marked epoch is the one that was actually
      checkpointed.
    * Metric space is **normalized**, labelled as such on the axis. These curves
      are never co-plotted with the Kelvin eval metrics; ``figures.yaml`` pins
      ``metric_space: normalized`` and ``stats.assert_same_space`` enforces it.
    * Seed spread is drawn as individual lines, not a band, until three seeds
      exist -- a band over two seeds is a fiction.
    """
    blocked(requirement,
            "needs multi-seed train_metrics.csv for forcing, source, "
            "source_itr and interfaces; no canonical training run survives in "
            "this workspace.",
            key="F03_learning_curves")


__all__ = ["learning_curves"]
