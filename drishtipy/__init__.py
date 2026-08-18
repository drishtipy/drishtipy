"""
drishtipy
============

A lightweight pandas DataFrame profiler that generates schema,
statistics, data-quality, ML-readiness, and ETL-readiness reports.
"""

import pandas as pd

from .core import DataProfiler
from .compare import DataComparator
from .accessor import ProfileAccessor
from .relationships import RelationshipAnalyzer, RelationshipConfig, RelationshipResult
from . import accessor  # noqa: F401  (import registers df.profile accessor)

__version__ = "0.6.0"
__all__ = [
    "DataProfiler",
    "DataComparator",
    "ProfileAccessor",
    "profile_of",
    "RelationshipAnalyzer",
    "RelationshipConfig",
    "RelationshipResult",
]


def profile_of(df: pd.DataFrame) -> ProfileAccessor:
    """
    Type-checker-friendly alternative to ``df.profile``.

    ``df.profile`` works perfectly at runtime — it's a pandas
    accessor registered via ``register_dataframe_accessor``. But
    static type checkers (Pyright/Pylance in VSCode, mypy) have no
    way to see that dynamic registration, because it isn't declared
    in pandas' own type stubs. Pandas' stubs still need to type
    *some* return value for arbitrary attribute access — since
    ``df.some_column`` is valid pandas syntax — so they fall back to
    ``DataFrame.__getattr__(self, name: str) -> Series[Any]``. That's
    the "(dynamic attribute) profile: Series[Any]" you see on hover:
    it's pandas' generic column-access stub, not anything specific to
    drishtipy, and it means your editor won't autocomplete or
    type-check ``df.profile.<method>(...)`` calls.

    This has no effect on runtime behavior — the code still runs
    correctly either way. It's purely a static-analysis/autocomplete
    limitation shared by every third-party pandas accessor package
    (this isn't specific to drishtipy).

    ``profile_of(df)`` returns the exact same accessor object, just
    with an explicit ``ProfileAccessor`` return type annotation, so
    your editor gets full autocomplete and type checking:

        from drishtipy import profile_of

        profile_of(df).correlations(by="pairs")
        profile_of(df).alerts(min_severity="high")

    is exactly equivalent to:

        df.profile.correlations(by="pairs")
        df.profile.alerts(min_severity="high")

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    ProfileAccessor
    """
    return df.profile  # type: ignore[return-value]
