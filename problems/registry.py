from __future__ import annotations

from problems.base import ProblemSpec
from problems.forcing import ForcingProblem
from problems.forcing_itr_sin import ForcingItrSinProblem
from problems.interfaces import InterfacesProblem
from problems.source import SourceProblem
from problems.source_itr_sin import SourceItrSinProblem

# Benchmarks register here as constructors taking a representation string.
REGISTRY: dict[str, type[ProblemSpec]] = {
    "forcing": ForcingProblem,
    "forcing_itr_sin": ForcingItrSinProblem,
    "interfaces": InterfacesProblem,
    "source": SourceProblem,
    "source_itr_sin": SourceItrSinProblem,
}

# The only public representation value. Kept as an explicit axis so run
# provenance (`config_used.yaml`, published manifests) stays readable.
REPRESENTATIONS: tuple[str, ...] = ("temporal_encoder",)


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
