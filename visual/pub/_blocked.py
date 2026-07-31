"""One way for a figure to say it cannot be drawn yet.

A blocked figure raises. It never returns an empty axes, a placeholder, or a
figure drawn from whatever happened to be lying around -- that is the same
silent-skip failure mode as ``visual/_common.py:_print_skip`` and it is exactly
what this package exists to prevent.

Under the default ``--strict`` these figures normally fail earlier, in
:func:`visual.pub.manifest.Manifest.resolve`, with a ``ProvenanceError`` naming
the missing artifact. :func:`blocked` is the second gate: it fires under
``--allow-missing``, where provenance has been downgraded to a degradation and
the drawing function is actually entered.
"""

from __future__ import annotations

from typing import NoReturn

VERIFY_HINT = "`python -m visual.pub --verify` prints the regeneration command."


def blocked(requirement, detail: str, *, key: str = "") -> NoReturn:
    """Raise naming the figure, its unmet requirement sets, and the way out.

    ``requirement`` is the :class:`~visual.pub.manifest.FigureRequirement` that
    ``render`` passes in; ``key`` is only needed when a drawing function is
    called directly, outside ``render``.
    """
    key = key or getattr(requirement, "key", "") or "<unknown figure>"
    requires = tuple(getattr(requirement, "requires", ()) or ())
    needs = ", ".join(requires) if requires else "no declared requirement set"
    raise NotImplementedError(f"{key}: {detail} Requires: {needs}. {VERIFY_HINT}")


__all__ = ["VERIFY_HINT", "blocked"]
