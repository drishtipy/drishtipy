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

Every method below carries a full docstring (parameters, return
shape, examples) so that IDE tooltips and Jupyter's inspector
(``df.profile.alerts?`` or Shift+Tab) show complete, accurate
signature help without needing to look up DataProfiler directly.
"""

from __future__ import annotations

import pandas as pd

from .core import DataProfiler
from .compare import DataComparator


@pd.api.extensions.register_dataframe_accessor("profile")
class ProfileAccessor:
    """
    ``df.profile`` accessor — profiling and comparison shortcuts for
    any pandas DataFrame, backed by :class:`~drishtipy.DataProfiler`
    and :class:`~drishtipy.DataComparator`.
    """

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
        """
        Full profiling report: Schema, Statistics, Quality, ML, and
        ETL sections in one call.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).
            Restricts profiling to columns of that dtype family.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Keys: ``"Schema"``, ``"Statistics"``, ``"Quality"``,
            ``"ML"``, ``"ETL"``.

        Raises
        ------
        ValueError
            If ``column_type`` is not a recognized value.

        Examples
        --------
        >>> df.profile.info()
        >>> df.profile.info(column_type="numeric")
        """
        return self._p.info_dataframe(section="All", column_type=column_type)

    def schema(self, column_type: str = "All") -> pd.DataFrame:
        """
        Per-column schema: dtype, null count/%, unique count/%, and
        memory usage.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> df.profile.schema()
        """
        return self._p.info_dataframe(section="schema", column_type=column_type)

    def statistics(self, column_type: str = "All") -> pd.DataFrame:
        """
        Descriptive statistics per column — numeric summary stats
        (mean, std, quantiles, etc.) and categorical mode/frequency.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> df.profile.statistics()
        """
        return self._p.info_dataframe(section="statistics", column_type=column_type)

    def quality(self, column_type: str = "All") -> pd.DataFrame:
        """
        Per-column data-quality metrics: duplicates, missing values,
        zero counts, and outlier counts.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> df.profile.quality()
        """
        return self._p.info_dataframe(section="quality", column_type=column_type)

    def quality_score(
        self,
        column_type: str = "All",
        by: str = "overall",
        weights: dict | None = None,
    ) -> pd.DataFrame:
        """
        Composite data-quality score (0-100, higher is better) across
        five dimensions: Missing Values, Duplicates, Outliers, Data
        Types, and Invalid Values.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).
            Restricts scoring to columns of that dtype family.
        by : str, default "overall"
            "overall" returns a single blended score plus each of the
            five dimension scores. "column" returns one row per
            column with its own dimension scores and a per-column
            Quality Score.
        weights : dict, optional
            Override the relative weight of each dimension in the
            blended score. Keys: "missing", "duplicates", "outliers",
            "dtypes", "invalid". Any keys you omit keep their default
            weight (20 each); values don't need to sum to 100, they
            are normalized automatically.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If ``by`` or ``column_type`` isn't recognized, if
            ``weights`` has an unknown key, or if the weights sum to
            zero or less.

        Examples
        --------
        >>> df.profile.quality_score()
        >>> df.profile.quality_score(by="column")
        >>> df.profile.quality_score(weights={"missing": 40, "outliers": 30})
        """
        return self._p.quality_score(column_type=column_type, by=by, weights=weights)

    def pii(
        self,
        min_confidence: float = 60.0,
        sample: int | None = 2000,
        mask: bool = False,
        mask_char: str = "*",
    ):
        """
        Detect columns that look like personally identifiable
        information — EMAIL, PHONE, POSSIBLE_ID (Aadhaar-like),
        POSSIBLE_PAN, IP, POSSIBLE_NAME, POSSIBLE_ADDRESS — using
        pattern matching plus a column-name hint.

        Parameters
        ----------
        min_confidence : float, default 60.0
            Only report (or mask) columns whose best-matching PII
            type reaches at least this confidence (0-100).
        sample : int, optional, default 2000
            Cap the number of non-null values inspected per column,
            for speed on large DataFrames. Pass ``None`` to scan every
            value.
        mask : bool, default False
            If False (default), return a report DataFrame of detected
            columns. If True, return a **copy of the DataFrame** with
            values in detected PII columns masked, instead of a
            report.
        mask_char : str, default "*"
            Character used to redact masked characters when
            ``mask=True``.

        Returns
        -------
        pandas.DataFrame
            A report with columns ``Column``, ``PII Type``,
            ``Confidence %`` (sorted by confidence, descending) when
            ``mask=False``; otherwise a masked copy of the DataFrame.

        Raises
        ------
        ValueError
            If ``min_confidence`` is not between 0 and 100.

        Examples
        --------
        >>> df.profile.pii()
        >>> df.profile.pii(mask=True)
        >>> df.profile.pii(min_confidence=80, sample=None)
        """
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
        """
        Correlation between numeric columns — useful for spotting
        redundant features or multicollinearity before modeling.

        Parameters
        ----------
        method : str, default "pearson"
            One of {"pearson", "spearman", "kendall"}
            (case-insensitive). Forwarded to ``DataFrame.corr``.
        by : str, default "matrix"
            "matrix" returns the full column-by-column correlation
            matrix. "pairs" returns a tidy DataFrame — one row per
            column pair — sorted by correlation strength (descending,
            by absolute value).
        threshold : float, optional
            Only used when ``by="pairs"``. Keep only pairs whose
            absolute correlation is at least this value, e.g. ``0.8``
            to surface likely multicollinearity candidates.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If ``method`` or ``by`` isn't recognized, or fewer than
            two numeric columns are available.

        Examples
        --------
        >>> df.profile.correlations()
        >>> df.profile.correlations(by="pairs", threshold=0.8)
        >>> df.profile.correlations(method="spearman")
        """
        return self._p.correlations(method=method, by=by, threshold=threshold)

    def alerts(
        self,
        column_type: str = "All",
        min_severity: str = "low",
    ) -> pd.DataFrame:
        """
        Single, severity-ranked list of data-quality and risk
        warnings — missing data, duplicate rows, outliers,
        constant/near-constant columns, skew, high cardinality,
        columns stored as the wrong dtype, messy text, highly
        correlated numeric pairs, and potential PII.

        A convenience layer over the other sections (``quality``,
        ``ml``, ``etl``, ``correlations``, ``pii``): it doesn't
        compute anything those don't already compute, it just
        collects the noteworthy findings into one ranked table.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).
        min_severity : str, default "low"
            Minimum severity to include: "high" (High only), "medium"
            (High + Medium), or "low" (everything). Case-insensitive.

        Returns
        -------
        pandas.DataFrame
            Columns: ``Severity``, ``Column``, ``Alert``, ``Details``,
            sorted High -> Medium -> Low, then by column. Table-wide
            issues (duplicate rows, correlated pairs) use
            ``"(table)"`` as the Column value. Empty if nothing was
            flagged.

        Raises
        ------
        ValueError
            If ``column_type`` or ``min_severity`` isn't recognized.

        Examples
        --------
        >>> df.profile.alerts()
        >>> df.profile.alerts(min_severity="high")
        """
        return self._p.alerts(column_type=column_type, min_severity=min_severity)

    def ml(self, column_type: str = "All") -> pd.DataFrame:
        """
        ML-readiness report per column: feature type, suggested
        encoding/scaling, and a keep/drop/review status.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> df.profile.ml()
        """
        return self._p.info_dataframe(section="ml", column_type=column_type)

    def etl(self, column_type: str = "All") -> pd.DataFrame:
        """
        ETL-readiness report per column: data issues found (e.g.
        stray whitespace, mixed types, special characters) and a
        recommended cleaning action.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> df.profile.etl()
        """
        return self._p.info_dataframe(section="etl", column_type=column_type)

    def html(
        self,
        path: str | None = None,
        section: str = "All",
        column_type: str = "All",
        title: str = "Data Profile Report",
    ) -> str:
        """
        Render the profiling report as a standalone HTML page.

        Parameters
        ----------
        path : str, optional
            If given, writes the HTML to this file path.
        section : str, default "All"
            One of {"All", "Schema", "Statistics", "Quality", "ML", "ETL"}
            (case-insensitive).
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).
        title : str, default "Data Profile Report"
            Page title shown at the top of the report.

        Returns
        -------
        str
            The rendered HTML (also written to ``path`` if provided).

        Examples
        --------
        >>> df.profile.html("report.html")
        >>> html_text = df.profile.html()  # no file written
        """
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
        """
        Compare this DataFrame (as "before") against ``other`` (as
        "after"): dataset-level metrics, schema changes, and
        column-level quality/ML stats, side by side.

        Parameters
        ----------
        other : pandas.DataFrame
            The updated DataFrame to compare against ``self``.
        level : str, default "all"
            One of {"All", "Dataset", "Schema", "Quality", "ML"}
            (case-insensitive):

            - "dataset": row/column counts, missing cells, duplicate
              rows, memory usage.
            - "schema": added/removed columns and dtype changes.
            - "quality" or "ml": per-column missing counts, zero
              counts, descriptive statistics, quantiles, outliers
              (numeric columns), and unique-value counts (categorical
              columns).
            - "all": every section above, concatenated.
        changes_only : bool, default False
            If True, drop rows where ``Before`` and ``After`` are
            identical. Schema rows (added/removed/dtype-changed
            columns) are always kept.
        columns : list of str, optional
            Restrict the comparison to these columns only. Raises
            ``ValueError`` if a name isn't present in either
            DataFrame.

        Returns
        -------
        pandas.DataFrame
            Long-format comparison with columns ``Section``, ``Metric``,
            ``Before``, ``After``, ``Change``, and ``Status``.

        Raises
        ------
        TypeError
            If ``other`` is not a pandas DataFrame.
        ValueError
            If ``level`` is not a recognized value, or ``columns``
            contains a name not present in either DataFrame.

        Examples
        --------
        >>> df.profile.compare(new_df)
        >>> df.profile.compare(new_df, level="schema")
        >>> df.profile.compare(new_df, changes_only=True)
        """
        return self._comparator.compare_dataframe(
            self._df, other, level=level, changes_only=changes_only, columns=columns
        )

    def compare_summary(
        self, other: pd.DataFrame, columns: list | None = None
    ) -> pd.DataFrame:
        """
        A compact, one-row-per-column overview of what changed
        between this DataFrame and ``other``.

        Unlike ``compare`` (long-format, many rows per column), this
        returns exactly one row per added/removed/common column with
        a short human-readable summary of the difference — much
        easier to eyeball for wide DataFrames.

        Parameters
        ----------
        other : pandas.DataFrame
            The updated DataFrame to compare against ``self``.
        columns : list of str, optional
            Restrict the summary to these columns only. Raises
            ``ValueError`` if a name isn't present in either
            DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns: ``Column``, ``Status``, ``Summary``.

        Examples
        --------
        >>> df.profile.compare_summary(new_df)
        """
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
        """
        Render a before/after comparison against ``other`` as a
        standalone HTML page.

        Parameters
        ----------
        other : pandas.DataFrame
            The updated DataFrame to compare against ``self``.
        path : str, optional
            If given, writes the HTML to this file path.
        level, changes_only, columns : same meaning as in ``compare``.
        title : str, default "Data Comparison Report"
            Page title shown at the top of the report.

        Returns
        -------
        str
            The rendered HTML (also written to ``path`` if provided).

        Examples
        --------
        >>> df.profile.compare_html(new_df, path="compare.html")
        """
        return self._comparator.to_html(
            self._df,
            other,
            path=path,
            level=level,
            changes_only=changes_only,
            columns=columns,
            title=title,
        )
