"""
Registers a `.profile` accessor on every pandas DataFrame, so you can
call it the same way you'd call `df.info()` or `df.describe()` —
no need to wrap the DataFrame in DataProfiler/DataComparator yourself.

Importing `drishtipy` (or `drishtipy.accessor`) registers the
accessor as a side effect:

    import pandas as pd
    import drishtipy  # registers df.profile

    df = pd.DataFrame(...)
    df.profile.info()             # same as DataProfiler(df).info_dataframe()
    df.profile.schema()
    df.profile.statistics()
    df.profile.quality()
    df.profile.quality_score()
    df.profile.quality_score(by="column")
    df.profile.pii()
    df.profile.pii(mask=True)
    df.profile.correlations()
    df.profile.correlations(by="pairs", threshold=0.8)
    df.profile.alerts()
    df.profile.ml()
    df.profile.etl()
    df.profile.html("report.html")

    df.profile.compare(other_df)          # same as DataComparator().compare_dataframe(df, other_df)
    df.profile.compare_summary(other_df)
    df.profile.compare_html(other_df, path="compare.html")
"""

from __future__ import annotations

import pandas as pd

from .core import DataProfiler
from .compare import DataComparator


@pd.api.extensions.register_dataframe_accessor("profile")
class ProfileAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._df = pandas_obj
        self._profiler: DataProfiler | None = None
        self._comparator = DataComparator()

    @property
    def _p(self) -> DataProfiler:
        # Built lazily, and rebuilt if the underlying DataFrame object
        # is swapped out (e.g. df = df.copy() reassigned to the same name).
        if self._profiler is None or self._profiler.df is not self._df:
            self._profiler = DataProfiler(self._df)
        return self._profiler

    # -------------------------------------------------
    # Profiling — mirrors DataProfiler.info_dataframe
    # -------------------------------------------------

    def info(self, column_type: str = "All"):
        """Full report: dict with Schema, Statistics, Quality, ML, ETL."""
        return self._p.info_dataframe(section="All", column_type=column_type)

    def schema(self, column_type: str = "All") -> pd.DataFrame:
        return self._p.info_dataframe(section="schema", column_type=column_type)

    def statistics(self, column_type: str = "All") -> pd.DataFrame:
        return self._p.info_dataframe(section="statistics", column_type=column_type)

    def quality(self, column_type: str = "All") -> pd.DataFrame:
        return self._p.info_dataframe(section="quality", column_type=column_type)

    def quality_score(
        self,
        column_type: str = "All",
        by: str = "overall",
        weights: dict | None = None,
    ) -> pd.DataFrame:
        """Composite quality score. See DataProfiler.quality_score."""
        return self._p.quality_score(column_type=column_type, by=by, weights=weights)

    def pii(
        self,
        min_confidence: float = 60.0,
        sample: int | None = 2000,
        mask: bool = False,
        mask_char: str = "*",
    ):
        """Detect (or mask) columns that look like PII. See DataProfiler.pii."""
        return self._p.pii(
            min_confidence=min_confidence,
            sample=sample,
            mask=mask,
            mask_char=mask_char,
        )

    def correlations(
        self,
        method: str = "pearson",
        by: str = "matrix",
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Numeric correlations. See DataProfiler.correlations."""
        return self._p.correlations(method=method, by=by, threshold=threshold)

    def alerts(
        self,
        column_type: str = "All",
        min_severity: str = "low",
    ) -> pd.DataFrame:
        """Severity-ranked list of data-quality warnings. See DataProfiler.alerts."""
        return self._p.alerts(column_type=column_type, min_severity=min_severity)

    def ml(self, column_type: str = "All") -> pd.DataFrame:
        return self._p.info_dataframe(section="ml", column_type=column_type)

    def etl(self, column_type: str = "All") -> pd.DataFrame:
        return self._p.info_dataframe(section="etl", column_type=column_type)

    def html(
        self,
        path: str | None = None,
        section: str = "All",
        column_type: str = "All",
        title: str = "Data Profile Report",
    ) -> str:
        return self._p.to_html(
            path=path, section=section, column_type=column_type, title=title
        )

    # -------------------------------------------------
    # Comparison — mirrors DataComparator
    # -------------------------------------------------

    def compare(
        self,
        other: pd.DataFrame,
        level: str = "all",
        changes_only: bool = False,
        columns: list | None = None,
    ) -> pd.DataFrame:
        """Compare this DataFrame (as 'before') against `other` (as 'after')."""
        return self._comparator.compare_dataframe(
            self._df, other, level=level, changes_only=changes_only, columns=columns
        )

    def compare_summary(
        self, other: pd.DataFrame, columns: list | None = None
    ) -> pd.DataFrame:
        return self._comparator.summary(self._df, other, columns=columns)

    def compare_html(
        self,
        other: pd.DataFrame,
        path: str | None = None,
        level: str = "all",
        changes_only: bool = False,
        columns: list | None = None,
        title: str = "Data Comparison Report",
    ) -> str:
        return self._comparator.to_html(
            self._df,
            other,
            path=path,
            level=level,
            changes_only=changes_only,
            columns=columns,
            title=title,
        )
