"""Publication figure package.

Every figure here is rendered from artifacts whose provenance is resolved,
hashed, and written to a sidecar. A figure that cannot name its sources does
not render.

Submodules are imported lazily so ``python -m visual.pub --list`` stays fast
and does not pull in torch.
"""

from __future__ import annotations

__all__ = ["FIGURES", "FigureSpec", "get_figure", "render", "render_many"]


def __getattr__(name: str):
    if name in __all__:
        from visual.pub import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
