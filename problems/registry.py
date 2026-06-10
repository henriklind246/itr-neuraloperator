from __future__ import annotations

from problems.base import ProblemSpec
from problems.forcing import ForcingProblem
from problems.interfaces import InterfacesProblem
from problems.source import SourceProblem

# Benchmarks register here as constructors taking a representation string.
REGISTRY: dict[str, type[ProblemSpec]] = {
    "forcing": ForcingProblem,
    "interfaces": InterfacesProblem,
    "source": SourceProblem,
}

# The two public representation values. Every benchmark supports both.
REPRESENTATIONS: tuple[str, ...] = ("temporal_encoder", "bins")


def get_problem(name: str, representation: str = "temporal_encoder") -> ProblemSpec:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(REGISTRY)}"
        ) from None
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r} for benchmark {name!r}. "
            f"Available: {list(REPRESENTATIONS)}"
        )
    return cls(representation)
