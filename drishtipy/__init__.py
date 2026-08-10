"""
drishtipy
============

A lightweight pandas DataFrame profiler that generates schema,
statistics, data-quality, ML-readiness, and ETL-readiness reports.
"""

from .core import DataProfiler
from .compare import DataComparator
from . import accessor  # noqa: F401  (import registers df.profile accessor)

__version__ = "0.5.1"
__all__ = ["DataProfiler", "DataComparator"]
