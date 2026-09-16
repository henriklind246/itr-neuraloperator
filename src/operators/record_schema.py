"""Column layout of ``test_records.csv``, shared by the writer and the readers.

This lives apart from ``src/operators/eval.py`` because the publication
analysis path only ever needs the names. Importing them from the eval module
pulled in torch, the dataset, and the 1D solver's scipy dependency, which made
``python -m visual.pub`` unrunnable on a cluster that has the records and
pandas but no training stack.

Keep this module dependency-free.
"""

from __future__ import annotations

TEST_RECORD_FIELDS = [
    "provenance_id", "sim_id", "s", "j", "t_s", "t_bar", "R_c", "benchmark",
    "temporal_family", "spatial_family",
    "x_h", "y_h", "A", "freq", "regime",
    "R_c_A",
    "x_I", "rel_l2_pct", "iface_rel_l2_pct",
    "nrmse_pct", "rmse_K", "gnrmse_pct",
    "node_jump_rmse_K", "node_jump_nrmse_pct", "node_jump_gnrmse_pct",
    "node_jump_abs_max_pred_K", "node_jump_abs_max_true_K",
    # OOD identity (joined from ood_metadata.jsonl by sim_id; in-distribution
    # defaults when the sidecar is absent).
    "ood_axis", "ood_value", "ood_repeat", "latents_hash", "distribution_class",
    # Time-pair protocol tagging (from dataset._pair_tags; empty without protocols).
    "protocol",
    "source_time_requested", "source_time_actual", "lead_time_actual",
    "target_time_requested", "target_time_actual",
    "dataset_t_final", "time_norm_horizon",
    # Pooled sufficient statistics (physical K^2) so the aggregator reconstructs
    # RMSE_sim and pooled rel-L2 without re-reading trajectories.
    "sse_K2", "num_error_cells",
    "interface_sse_K2", "num_interface_cells",
    "target_sse_K2", "interface_target_sse_K2",
]

__all__ = ["TEST_RECORD_FIELDS"]
