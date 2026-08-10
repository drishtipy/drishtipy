"""
Comparison module for drishtipy.

Contains the DataComparator class, which diffs two pandas DataFrames
(e.g. "before" and "after" a cleaning/ETL step) across dataset-level
metrics, schema changes, missing/zero counts, descriptive statistics,
quantiles, outliers, and category cardinality.
"""

from __future__ import annotations

import pandas as pd

from .report import render_html_report

__all__ = ["DataComparator"]

_VALID_LEVELS = {"all", "dataset", "schema", "quality", "ml"}

_NUMERIC_STATS = ["mean", "median", "std", "min", "max"]
_QUANTILES = [0.25, 0.50, 0.75, 0.95]


def _iqr_outlier_count(x: pd.Series) -> int:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    return int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())


def _select_columns(before: pd.DataFrame, after: pd.DataFrame, columns):
    columns = list(columns)

    unknown = [c for c in columns if c not in before.columns and c not in after.columns]
    if unknown:
        raise ValueError(
            f"Column(s) not found in either DataFrame: {unknown}"
        )

    before_cols = [c for c in columns if c in before.columns]
    after_cols = [c for c in columns if c in after.columns]

    return before[before_cols], after[after_cols]


class DataComparator:
    """
    Compares two pandas DataFrames — typically a "before" and "after"
    snapshot around a cleaning, transformation, or ETL step — and
    reports what changed.

    Examples
    --------
    >>> import pandas as pd
    >>> from drishtipy import DataComparator
    >>> before = pd.DataFrame({"a": [1, 2, None, 4]})
    >>> after = pd.DataFrame({"a": [1, 2, 3, 4]})
    >>> DataComparator().compare_dataframe(before, after, level="quality")
    """

    def compare_dataframe(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        level: str = "all",
        changes_only: bool = False,
        columns: list | None = None,
    ) -> pd.DataFrame:
        """
        Compare ``before`` and ``after`` DataFrames.

        Parameters
        ----------
        before : pandas.DataFrame
            The original/baseline DataFrame.
        after : pandas.DataFrame
            The updated DataFrame to compare against ``before``.
        level : str, default "all"
            One of {"All", "Dataset", "Schema", "Quality", "ML"}
            (case-insensitive):

            - "dataset": row/column counts, missing cells, duplicate
              rows, memory usage.
            - "schema": added/removed columns and dtype changes.
            - "quality" or "ml": per-column missing counts, zero
              counts, descriptive statistics, quantiles, outliers
              (numeric columns), and unique-value counts (categorical
              columns). "quality" and "ml" currently produce the same
              column-level report.
            - "all": every section above, concatenated.
        changes_only : bool, default False
            If True, drop rows where ``Before`` and ``After`` are
            identical (e.g. every "Same" dataset metric, every
            unchanged statistic/quantile/outlier count). Schema rows
            (added/removed/dtype-changed columns) are always kept
            since they only appear when something differs. Useful for
            cutting a long report down to just what actually changed.
        columns : list of str, optional
            Restrict the comparison to these columns only. A column
            missing from one side is treated as added/removed, same as
            with the full DataFrame. Raises ``ValueError`` if a name
            isn't present in either DataFrame. Note: with this set,
            "dataset"-level row/column counts still reflect the full
            ``before``/``after`` DataFrames, not just the selection.

        Returns
        -------
        pandas.DataFrame
            Long-format comparison with columns ``Section``, ``Metric``,
            ``Before``, ``After``, ``Change``, and ``Status`` (not every
            row populates every column).

        Raises
        ------
        TypeError
            If ``before`` or ``after`` is not a pandas DataFrame.
        ValueError
            If ``level`` is not a recognized value, or ``columns``
            contains a name not present in either DataFrame.
        """

        if not isinstance(before, pd.DataFrame):
            raise TypeError(
                f"before must be a pandas DataFrame, got {type(before).__name__}"
            )

        if not isinstance(after, pd.DataFrame):
            raise TypeError(
                f"after must be a pandas DataFrame, got {type(after).__name__}"
            )

        level = level.lower()

        if level not in _VALID_LEVELS:
            raise ValueError(
                f"Invalid level '{level}'. Must be one of {sorted(_VALID_LEVELS)}."
            )

        full_before, full_after = before, after
        if columns is not None:
            before, after = _select_columns(before, after, columns)

        results = []

        if level in ("all", "dataset", "quality"):
            results.extend(self._dataset_level(full_before, full_after))

        if level in ("all", "schema"):
            results.extend(self._schema_level(before, after))

        if level in ("all", "quality", "ml"):
            results.extend(self._column_level(before, after))

        report = pd.DataFrame(results)

        if changes_only and not report.empty:
            report = self._filter_changes(report)

        return report

    def _filter_changes(self, report: pd.DataFrame) -> pd.DataFrame:
        is_schema = report["Section"] == "Schema"

        if "Change" in report.columns:
            changed = report["Change"].fillna(0) != 0
        else:
            changed = pd.Series(False, index=report.index)

        return report[is_schema | changed].reset_index(drop=True)

    def summary(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        columns: list | None = None,
    ) -> pd.DataFrame:
        """
        A compact, one-row-per-column overview of what changed.

        Unlike ``compare_dataframe`` (long-format, many rows per
        column), this returns exactly one row per added/removed/common
        column with a short human-readable summary of the difference —
        much easier to eyeball for wide DataFrames.

        Parameters
        ----------
        before : pandas.DataFrame
        after : pandas.DataFrame
        columns : list of str, optional
            Restrict the summary to these columns only. Raises
            ``ValueError`` if a name isn't present in either DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns: ``Column``, ``Status``, ``Summary``.
        """

        if not isinstance(before, pd.DataFrame):
            raise TypeError(
                f"before must be a pandas DataFrame, got {type(before).__name__}"
            )

        if not isinstance(after, pd.DataFrame):
            raise TypeError(
                f"after must be a pandas DataFrame, got {type(after).__name__}"
            )

        if columns is not None:
            before, after = _select_columns(before, after, columns)

        rows = []

        old_cols = set(before.columns)
        new_cols = set(after.columns)

        for c in sorted(old_cols - new_cols):
            rows.append(
                {"Column": c, "Status": "Removed", "Summary": "Column removed"}
            )

        for c in sorted(new_cols - old_cols):
            rows.append(
                {"Column": c, "Status": "Added", "Summary": "Column added"}
            )

        for col in sorted(old_cols & new_cols):
            b = before[col]
            a = after[col]

            notes = []

            old_dtype, new_dtype = str(b.dtype), str(a.dtype)
            if old_dtype != new_dtype:
                notes.append(f"dtype {old_dtype}->{new_dtype}")

            bnull, anull = int(b.isna().sum()), int(a.isna().sum())
            if bnull != anull:
                notes.append(f"missing {bnull}->{anull}")

            if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
                bo, ao = _iqr_outlier_count(b), _iqr_outlier_count(a)
                if bo != ao:
                    notes.append(f"outliers {bo}->{ao}")

                bmean, amean = b.mean(), a.mean()
                if (
                    pd.notna(bmean)
                    and pd.notna(amean)
                    and round(bmean, 4) != round(amean, 4)
                ):
                    notes.append(f"mean {round(bmean, 2)}->{round(amean, 2)}")
            else:
                bu, au = b.nunique(), a.nunique()
                if bu != au:
                    notes.append(f"unique {bu}->{au}")

            status = "Changed" if notes else "Unchanged"
            summary = "; ".join(notes) if notes else "No change"

            rows.append({"Column": col, "Status": status, "Summary": summary})

        return pd.DataFrame(rows)

    # =====================================================
    # DATASET LEVEL
    # =====================================================

    def _dataset_level(self, before: pd.DataFrame, after: pd.DataFrame):
        rows = []

        metrics = {
            "Rows": (len(before), len(after)),
            "Columns": (len(before.columns), len(after.columns)),
            "Missing Cells": (
                int(before.isna().sum().sum()),
                int(after.isna().sum().sum()),
            ),
            "Duplicate Rows": (
                int(before.duplicated().sum()),
                int(after.duplicated().sum()),
            ),
            "Memory MB": (
                round(before.memory_usage(deep=True).sum() / 1024**2, 2),
                round(after.memory_usage(deep=True).sum() / 1024**2, 2),
            ),
        }

        for name, (b, a) in metrics.items():
            if a < b:
                status = "Improved"
            elif a > b:
                status = "Increased"
            else:
                status = "Same"

            rows.append(
                {
                    "Section": "Dataset",
                    "Metric": name,
                    "Before": b,
                    "After": a,
                    "Change": a - b,
                    "Status": status,
                }
            )

        return rows

    # =====================================================
    # SCHEMA COMPARISON
    # =====================================================

    def _schema_level(self, before: pd.DataFrame, after: pd.DataFrame):
        rows = []

        old_cols = set(before.columns)
        new_cols = set(after.columns)

        for c in old_cols.union(new_cols):
            if c not in old_cols:
                rows.append(
                    {
                        "Section": "Schema",
                        "Metric": c,
                        "Before": "Missing",
                        "After": "Added",
                        "Status": "New Column",
                    }
                )

            elif c not in new_cols:
                rows.append(
                    {
                        "Section": "Schema",
                        "Metric": c,
                        "Before": "Present",
                        "After": "Removed",
                        "Status": "Deleted Column",
                    }
                )

            else:
                old_dtype = str(before[c].dtype)
                new_dtype = str(after[c].dtype)

                if old_dtype != new_dtype:
                    rows.append(
                        {
                            "Section": "Schema",
                            "Metric": c,
                            "Before": old_dtype,
                            "After": new_dtype,
                            "Status": "Dtype Changed",
                        }
                    )

        return rows

    # =====================================================
    # COLUMN LEVEL
    # =====================================================

    def _column_level(self, before: pd.DataFrame, after: pd.DataFrame):
        rows = []

        common = set(before.columns) & set(after.columns)

        for col in common:
            b = before[col]
            a = after[col]

            rows.extend(self._missing_row(col, b, a))

            if pd.api.types.is_numeric_dtype(b):
                rows.extend(self._zero_row(col, b, a))
                rows.extend(self._statistics_rows(col, b, a))
                rows.extend(self._quantile_rows(col, b, a))
                rows.extend(self._outlier_row(col, b, a))
            else:
                rows.extend(self._category_row(col, b, a))

        return rows

    def _missing_row(self, col, b: pd.Series, a: pd.Series):
        bnull = int(b.isna().sum())
        anull = int(a.isna().sum())

        return [
            {
                "Section": "Missing",
                "Metric": col,
                "Before": bnull,
                "After": anull,
                "Change": anull - bnull,
                "Status": "Fixed" if anull < bnull else "Same",
            }
        ]

    def _zero_row(self, col, b: pd.Series, a: pd.Series):
        bz = int((b == 0).sum())
        az = int((a == 0).sum())

        return [
            {
                "Section": "Zero",
                "Metric": col,
                "Before": bz,
                "After": az,
                "Change": az - bz,
                "Status": "Improved" if az < bz else "Same",
            }
        ]

    def _statistics_rows(self, col, b: pd.Series, a: pd.Series):
        rows = []

        for stat in _NUMERIC_STATS:
            bv = getattr(b, stat)()
            av = getattr(a, stat)()

            rows.append(
                {
                    "Section": "Statistics",
                    "Metric": f"{col}_{stat}",
                    "Before": round(float(bv), 4) if pd.notna(bv) else None,
                    "After": round(float(av), 4) if pd.notna(av) else None,
                    "Change": (
                        round(float(av - bv), 4)
                        if pd.notna(av) and pd.notna(bv)
                        else None
                    ),
                }
            )

        return rows

    def _quantile_rows(self, col, b: pd.Series, a: pd.Series):
        rows = []

        for q in _QUANTILES:
            bv = b.quantile(q)
            av = a.quantile(q)

            rows.append(
                {
                    "Section": "Quantile",
                    "Metric": f"{col}_Q{int(q * 100)}",
                    "Before": bv,
                    "After": av,
                    "Change": av - bv,
                }
            )

        return rows

    def _outlier_row(self, col, b: pd.Series, a: pd.Series):
        bo = _iqr_outlier_count(b)
        ao = _iqr_outlier_count(a)

        return [
            {
                "Section": "Outlier",
                "Metric": col,
                "Before": bo,
                "After": ao,
                "Change": ao - bo,
                "Status": "Reduced" if ao < bo else "Same",
            }
        ]

    def _category_row(self, col, b: pd.Series, a: pd.Series):
        bu = b.nunique()
        au = a.nunique()

        return [
            {
                "Section": "Category",
                "Metric": col,
                "Before": bu,
                "After": au,
                "Change": au - bu,
            }
        ]

    # =====================================================
    # HTML EXPORT
    # =====================================================

    def to_html(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        path: str | None = None,
        level: str = "all",
        changes_only: bool = False,
        columns: list | None = None,
        title: str = "Data Comparison Report",
    ) -> str:
        """
        Render a before/after comparison as a standalone HTML page.

        Parameters mirror ``compare_dataframe``, plus:

        path : str, optional
            If given, writes the HTML to this file path.
        title : str
            Page title shown at the top of the report.

        Returns
        -------
        str
            The rendered HTML (also written to ``path`` if provided).
        """

        report = self.compare_dataframe(
            before, after, level=level, changes_only=changes_only, columns=columns
        )
        summary_df = self.summary(before, after, columns=columns)

        sections = {"Summary": summary_df, "Comparison": report}
        html_text = render_html_report(sections, title=title)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_text)

        return html_text
