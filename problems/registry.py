from __future__ import annotations

from problems.base import ProblemSpec
from problems.forcing import ForcingProblem
from problems.interfaces import InterfacesProblem
from problems.source import SourceProblem

# Benchmarks register here.
REGISTRY: dict[str, ProblemSpec] = {
    "forcing": ForcingProblem(),
    "interfaces": InterfacesProblem(),
    "source": SourceProblem(),
}


def get_problem(name: str) -> ProblemSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(REGISTRY)}"
        ) from None
