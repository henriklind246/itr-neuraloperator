"""Benchmark problem adapters.

Each benchmark (forcing, interfaces, source) is expressed as a `ProblemSpec`
that owns its sampling, solver wiring, dataset-item construction, tensor dims,
schema validation, and diagnostics. The core pipeline (dataset, model, train,
eval, generate_dataset) selects one with `get_problem(name)` and stays
benchmark-agnostic.
"""

from problems.base import ProblemDims, ProblemSpec
from problems.registry import REGISTRY, get_problem

__all__ = ["ProblemDims", "ProblemSpec", "REGISTRY", "get_problem"]
