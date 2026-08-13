"""
Core module for drishtipy.

Contains the DataProfiler class, which generates schema, statistics,
data-quality, machine-learning readiness, and ETL-readiness reports
for a pandas DataFrame.
"""

from __future__ import annotations

import re

import pandas as pd
import numpy as np

from .report import render_html_report, ProfileReport

__all__ = ["DataProfiler"]

_VALID_SECTIONS = {"all", "schema", "statistics", "quality", "ml", "etl"}
_VALID_COLUMN_TYPES = {"all", "numeric", "categorical"}
_VALID_BY = {"overall", "column"}

_SECTION_MAP = {
    "ml": "ML",
    "schema": "Schema",
    "statistics": "Statistics",
    "quality": "Quality",
    "etl": "ETL",
}

# Default relative weights for blending the five quality_score()
# dimensions into a single Overall Quality Score. Don't need to sum
# to 100 — they're normalized internally — but are expressed that
# way here for readability.
_DEFAULT_QUALITY_WEIGHTS = {
    "missing": 20,
    "duplicates": 20,
    "outliers": 20,
    "dtypes": 20,
    "invalid": 20,
}

_VALID_CORR_METHODS = {"pearson", "spearman", "kendall"}
_VALID_CORR_BY = {"matrix", "pairs"}

_VALID_SEVERITIES = {"high", "medium", "low"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
# Correlation strength, in absolute value, used by alerts() to flag
# a numeric pair as a multicollinearity risk.
_ALERT_CORRELATION_THRESHOLD = 0.8

# =====================================================
# PII DETECTION HELPERS
# =====================================================

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
# Aadhaar-like: 12 digits, optionally grouped in 4s ("1234 5678 9012").
_AADHAAR_RE = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")
# Indian PAN: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).
_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# Column-name keywords that nudge confidence up for the matching
# PII type — a column literally called "email" should win over a
# column of free text that merely contains a few email-shaped values.
_PII_NAME_HINTS = {
    "EMAIL": ("email", "mail", "e-mail"),
    "PHONE": ("phone", "mobile", "contact", "cell", "tel"),
    "POSSIBLE_ID": ("aadhaar", "aadhar", "uid", "uidai"),
    "POSSIBLE_PAN": ("pan",),
    "IP": ("ip_address", "ipaddr", "ip"),
    "POSSIBLE_NAME": ("name",),
    "POSSIBLE_ADDRESS": ("address", "addr", "street", "location"),
}

_PII_NAME_HINT_BONUS = 10.0
_DEFAULT_PII_MIN_CONFIDENCE = 60.0
_DEFAULT_PII_SAMPLE = 2000


def _is_phone(x: str) -> bool:
    """Loosely matches Indian mobile numbers and general phone numbers."""
    cleaned = re.sub(r"[\s\-\(\)]", "", x)

    if cleaned.startswith("+91"):
        cleaned = cleaned[3:]
    elif cleaned.startswith("91") and len(cleaned) == 12:
        cleaned = cleaned[2:]
    elif cleaned.startswith("+"):
        cleaned = cleaned[1:]

    if re.fullmatch(r"[6-9]\d{9}", cleaned):
        return True

    return bool(re.fullmatch(r"\d{10,13}", cleaned))


def _is_ip(x: str) -> bool:
    """Matches IPv4 addresses (each octet 0-255, no leading zeros)."""
    parts = x.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit():
            return False
        if len(p) > 1 and p[0] == "0":
            return False
        if int(p) > 255:
            return False
    return True


def _looks_like_name(x: str) -> bool:
    """Heuristic: 2-4 alphabetic words (first+last name shape), no digits.

    Deliberately excludes single-word values — a lone capitalized word
    is far more often a city, category, or brand than a person's name,
    and requiring at least two words cuts down on that false-positive
    class without a name-database to check against.
    """
    x = x.strip()
    if not x or len(x) > 40 or any(ch.isdigit() for ch in x):
        return False
    words = x.split()
    if not (2 <= len(words) <= 4):
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z'.\-]*", w) for w in words)


_ADDRESS_KEYWORDS = (
    "street", "st.", "road", "rd.", "lane", "ln.", "avenue", "ave",
    "nagar", "colony", "sector", "block", "apartment", "apt", "floor",
    "marg", "chowk", "circle", "layout", "society", "phase",
)


def _looks_like_address(x: str) -> bool:
    """Heuristic: contains digits plus an address keyword or comma-parts."""
    x = x.strip()
    if len(x) < 8:
        return False
    has_digit = any(ch.isdigit() for ch in x)
    if not has_digit:
        return False
    lower = x.lower()
    has_keyword = any(k in lower for k in _ADDRESS_KEYWORDS)
    has_comma = "," in x
    return has_keyword or has_comma


class DataProfiler:
    """
    Profiles a pandas DataFrame across several dimensions:

    - Schema: dtypes, null/unique counts, memory usage per column.
    - Statistics: descriptive statistics (numeric + categorical mode).
    - Quality: duplicates, missing values, zeros, outliers.
    - ML: feature typing, encoding/scaling suggestions, feature status.
    - ETL: per-column data issues and recommended cleaning actions.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to profile.

    Examples
    --------
    >>> import pandas as pd
    >>> from drishtipy import DataProfiler
    >>> df = pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "y", "z"]})
    >>> profiler = DataProfiler(df)
    >>> report = profiler.info_dataframe(section="schema")
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df must be a pandas DataFrame, got {type(df).__name__}"
            )
        self.df = df

        # Populated by from_csv() when the source was larger than the
        # requested sample; plain DataProfiler(df) leaves these as-is.
        self.total_rows_in_source: int | None = None
        self.is_sampled: bool = False

    # =====================================================
    # LARGE-DATA LOADING
    # =====================================================

    @classmethod
    def from_csv(
        cls,
        path: str,
        sample_size: int = 100_000,
        chunksize: int = 50_000,
        random_state: int | None = None,
        **read_csv_kwargs,
    ) -> "DataProfiler":
        """
        Build a DataProfiler from a (possibly very large) CSV without
        loading the whole file into memory at once.

        The file is read in chunks of ``chunksize`` rows twice:

        1. First pass: count total rows only (cheap, streaming).
        2. Second pass: if the file has more than ``sample_size`` rows,
           keep each row with probability ``sample_size / total_rows``
           (a random, roughly-uniform sample of that size) instead of
           the whole file. If the file is smaller than ``sample_size``,
           it's loaded in full.

        This trades exactness for the descriptive/statistics/ML
        sections (they run on a sample) in exchange for constant
        memory use regardless of file size. Exact row/column counts
        are still tracked via ``total_rows_in_source``.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        sample_size : int, default 100_000
            Maximum number of rows to keep in memory for profiling.
        chunksize : int, default 50_000
            Rows read per chunk during both passes.
        random_state : int, optional
            Seed for reproducible sampling.
        **read_csv_kwargs
            Forwarded to ``pandas.read_csv`` (e.g. ``usecols``, ``dtype``,
            ``sep``, ``parse_dates``).

        Returns
        -------
        DataProfiler
            Wrapping either the full file (if small) or a random
            sample of it. Check ``.is_sampled`` and
            ``.total_rows_in_source`` to see which happened.
        """

        total_rows = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_csv_kwargs):
            total_rows += len(chunk)

        if total_rows <= sample_size:
            df = pd.read_csv(path, **read_csv_kwargs)
            instance = cls(df)
            instance.total_rows_in_source = total_rows
            instance.is_sampled = False
            return instance

        frac = sample_size / total_rows
        rng = np.random.default_rng(random_state)

        parts = []
        kept = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_csv_kwargs):
            mask = rng.random(len(chunk)) < frac
            kept_chunk = chunk[mask]
            parts.append(kept_chunk)
            kept += len(kept_chunk)

        sample_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        instance = cls(sample_df)
        instance.total_rows_in_source = total_rows
        instance.is_sampled = True
        return instance

    # =====================================================
    # PUBLIC API
    # =====================================================

    def info_dataframe(
        self,
        section: str = "All",
        column_type: str = "All",
    ):
        """
        Generate a profiling report.

        Parameters
        ----------
        section : str, default "All"
            One of {"All", "Schema", "Statistics", "Quality", "ML", "ETL"}
            (case-insensitive). "All" returns a dict containing every
            section; any other value returns just that section's
            DataFrame.
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).
            Restricts profiling to columns of that dtype family.

        Returns
        -------
        ProfileReport or pandas.DataFrame
            A ``ProfileReport`` (a ``dict[str, DataFrame]`` subclass —
            same access patterns as a plain dict, but with a styled
            HTML repr for Jupyter and a compact analytical summary
            for the terminal) when ``section="All"``, otherwise a
            single DataFrame for the requested section.

        Raises
        ------
        ValueError
            If ``section`` or ``column_type`` is not a recognized value.
        """

        section = section.lower()
        column_type = column_type.lower()

        if section not in _VALID_SECTIONS:
            raise ValueError(
                f"Invalid section '{section}'. Must be one of "
                f"{sorted(_VALID_SECTIONS)}."
            )

        if column_type not in _VALID_COLUMN_TYPES:
            raise ValueError(
                f"Invalid column_type '{column_type}'. Must be one of "
                f"{sorted(_VALID_COLUMN_TYPES)}."
            )

        # -------------------------
        # Column Filter
        # -------------------------

        if column_type == "numeric":
            df = self.df.select_dtypes(include="number")

        elif column_type == "categorical":
            df = self.df.select_dtypes(include=["object", "category", "string"])

        else:
            df = self.df

        if df.shape[1] == 0:
            raise ValueError(
                f"No columns match column_type='{column_type}'."
            )

        result = {}

        if section in ("all", "schema"):
            result["Schema"] = self._schema_info(df)

        if section in ("all", "statistics"):
            result["Statistics"] = self._statistics_info(df)

        if section in ("all", "quality"):
            result["Quality"] = self._quality_info(df)

        if section in ("all", "ml"):
            result["ML"] = self._ml_info(df)

        if section in ("all", "etl"):
            result["ETL"] = self._etl_info(df)

        if section == "all":
            return ProfileReport(result, self, title="Data Profile Report")

        return result[_SECTION_MAP[section]]

    # =====================================================
    # HTML EXPORT
    # =====================================================

    def to_html(
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
        section, column_type : str
            Same meaning as in ``info_dataframe``.
        title : str
            Page title shown at the top of the report.

        Returns
        -------
        str
            The rendered HTML (also written to ``path`` if provided).
        """

        report = self.info_dataframe(section=section, column_type=column_type)
        sections = report if isinstance(report, dict) else {section.title(): report}

        subtitle = None
        if self.is_sampled and self.total_rows_in_source is not None:
            subtitle = (
                f"Based on a random sample of {len(self.df):,} rows "
                f"out of {self.total_rows_in_source:,} total rows in source."
            )

        html_text = render_html_report(sections, title=title, subtitle=subtitle)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_text)

        return html_text

    # =====================================================
    # SCHEMA
    # =====================================================

    def _schema_info(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for i, col in enumerate(df.columns):
            s = df[col]
            memory = s.memory_usage(deep=True)
            n = len(s)

            rows.append(
                {
                    "#": i,
                    "Column": col,
                    "Dtype": str(s.dtype),
                    "Non-Null Count": int(s.notna().sum()),
                    "Missing Count": int(s.isna().sum()),
                    "Missing %": round(s.isna().mean() * 100, 2),
                    "Unique Values": int(s.nunique()),
                    "Unique %": round(s.nunique() / n * 100, 2) if n else 0.0,
                    "Memory Usage": memory,
                }
            )

        return pd.DataFrame(rows)

    # =====================================================
    # STATISTICS
    # =====================================================

    def _statistics_info(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for col in df.columns:
            s = df[col]
            mode = s.mode()

            row = {
                "Column": col,
                "Dtype": str(s.dtype),
                "Count": int(s.count()),
                "Missing Count": int(s.isna().sum()),
                "Mean": None,
                "Median": None,
                "Std": None,
                "Min": None,
                "Q1": None,
                "Q2": None,
                "Q3": None,
                "95%": None,
                "99%": None,
                "Max": None,
                "Range": None,
                "IQR": None,
                "Skewness": None,
                "Kurtosis": None,
                "Mode": mode.iloc[0] if len(mode) else None,
                "Mode Frequency": (
                    int(s.value_counts().iloc[0]) if len(s.dropna()) else 0
                ),
                "Geometric Mean": None,
                "Harmonic Mean": None,
                "RMS": None,
                "CV %": None,
                "MAD": None,
                "Z-Outlier Count": None,
            }

            if pd.api.types.is_numeric_dtype(s):
                non_null = s.dropna()
                q1 = s.quantile(0.25)
                q2 = s.quantile(0.50)
                q3 = s.quantile(0.75)
                mean = s.mean()
                std = s.std()

                row.update(
                    {
                        "Mean": round(mean, 4),
                        "Median": round(s.median(), 4),
                        "Std": round(std, 4),
                        "Min": s.min(),
                        "Q1": q1,
                        "Q2": q2,
                        "Q3": q3,
                        "95%": s.quantile(0.95),
                        "99%": s.quantile(0.99),
                        "Max": s.max(),
                        "Range": s.max() - s.min(),
                        "IQR": q3 - q1,
                        "Skewness": round(s.skew(), 4),
                        "Kurtosis": round(s.kurt(), 4),
                    }
                )

                if len(non_null):
                    # Geometric / harmonic mean are only defined for
                    # strictly positive data — left as None otherwise
                    # rather than silently producing a wrong number.
                    if (non_null > 0).all():
                        row["Geometric Mean"] = round(
                            float(np.exp(np.log(non_null).mean())), 4
                        )
                        row["Harmonic Mean"] = round(
                            float(len(non_null) / (1.0 / non_null).sum()), 4
                        )

                    row["RMS"] = round(
                        float(np.sqrt((non_null ** 2).mean())), 4
                    )
                    row["MAD"] = round(
                        float((non_null - non_null.mean()).abs().mean()), 4
                    )

                    if pd.notna(mean) and mean != 0:
                        row["CV %"] = round(float(std / abs(mean) * 100), 4)

                    if len(non_null) > 1 and pd.notna(std) and std != 0:
                        z = (non_null - non_null.mean()) / std
                        row["Z-Outlier Count"] = int((z.abs() > 3).sum())
                    else:
                        row["Z-Outlier Count"] = 0

            rows.append(row)

        return pd.DataFrame(rows)

    # =====================================================
    # QUALITY
    # =====================================================

    def _quality_info(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for col in df.columns:
            s = df[col]
            n = len(s)

            row = {
                "Column": col,
                "Duplicate Count": int(s.duplicated().sum()),
                "Missing Count": int(s.isna().sum()),
                "Zero Count": None,
                "Outlier Count": None,
                "Outlier %": None,
            }

            if pd.api.types.is_numeric_dtype(s):
                row["Zero Count"] = int((s == 0).sum())

                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1

                out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()

                row["Outlier Count"] = int(out)
                row["Outlier %"] = round(out / n * 100, 2) if n else 0.0

            rows.append(row)

        return pd.DataFrame(rows)

    # =====================================================
    # QUALITY SCORE
    # =====================================================

    def quality_score(
        self,
        column_type: str = "All",
        by: str = "overall",
        weights: dict | None = None,
    ) -> pd.DataFrame:
        """
        Compute a composite data-quality score (0-100, higher is
        better) across five dimensions:

        - Missing Values : 100 minus each column's missing-value %.
        - Duplicates      : 100 minus the % of fully duplicated rows.
        - Outliers        : 100 minus each numeric column's IQR
          outlier %. Non-numeric columns score 100 (not applicable).
        - Data Types      : penalizes text columns that are actually
          numeric-looking (stored as the wrong dtype) or that mix
          more than one Python type among their non-null values.
        - Invalid Values  : for numeric columns, the % of infinite
          values; for text columns, the % of empty strings, stray
          leading/trailing whitespace, or unexpected special
          characters.

        Parameters
        ----------
        column_type : str, default "All"
            One of {"All", "Numeric", "Categorical"} (case-insensitive).
            Restricts scoring to columns of that dtype family.
        by : str, default "overall"
            "overall" returns a single blended score plus each of the
            five dimension scores. "column" returns one row per
            column with its own dimension scores and a per-column
            Quality Score. Note: at the column level, "Duplicates"
            measures repeated *values* within that column (as in the
            Quality section), not duplicate rows.
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
        """

        by = by.lower()
        column_type = column_type.lower()

        if by not in _VALID_BY:
            raise ValueError(
                f"Invalid by='{by}'. Must be one of {sorted(_VALID_BY)}."
            )

        if column_type not in _VALID_COLUMN_TYPES:
            raise ValueError(
                f"Invalid column_type '{column_type}'. Must be one of "
                f"{sorted(_VALID_COLUMN_TYPES)}."
            )

        if column_type == "numeric":
            df = self.df.select_dtypes(include="number")
        elif column_type == "categorical":
            df = self.df.select_dtypes(include=["object", "category", "string"])
        else:
            df = self.df

        if df.shape[1] == 0:
            raise ValueError(f"No columns match column_type='{column_type}'.")

        w = dict(_DEFAULT_QUALITY_WEIGHTS)
        if weights:
            unknown = set(weights) - set(w)
            if unknown:
                raise ValueError(
                    f"Unknown weight key(s) {sorted(unknown)}. Must be one "
                    f"of {sorted(w)}."
                )
            w.update(weights)

        total_w = sum(w.values())
        if total_w <= 0:
            raise ValueError("weights must sum to a positive number.")
        w = {k: v / total_w * 100 for k, v in w.items()}

        per_col = self._quality_score_by_column(df, w)

        if by == "column":
            return per_col

        n_rows = len(df)
        dup_row_pct = (
            round(df.duplicated().sum() / n_rows * 100, 2) if n_rows else 0.0
        )

        summary = {
            "Missing Values": round(per_col["Missing Score"].mean(), 2),
            "Duplicates": round(100 - dup_row_pct, 2),
            "Outliers": round(per_col["Outlier Score"].mean(), 2),
            "Data Types": round(per_col["Data Type Score"].mean(), 2),
            "Invalid Values": round(per_col["Invalid Score"].mean(), 2),
        }

        overall = round(
            (
                summary["Missing Values"] * w["missing"]
                + summary["Duplicates"] * w["duplicates"]
                + summary["Outliers"] * w["outliers"]
                + summary["Data Types"] * w["dtypes"]
                + summary["Invalid Values"] * w["invalid"]
            )
            / 100,
            2,
        )

        rows = [{"Metric": "Overall Quality Score", "Score": overall}]
        rows += [{"Metric": k, "Score": v} for k, v in summary.items()]

        return pd.DataFrame(rows)

    def _quality_score_by_column(self, df: pd.DataFrame, w: dict) -> pd.DataFrame:
        rows = []

        for col in df.columns:
            s = df[col]
            n = len(s)
            non_null = s.dropna()
            n_non_null = len(non_null)

            # ---- Missing ----
            missing_pct = round(s.isna().mean() * 100, 2) if n else 0.0
            missing_score = round(100 - missing_pct, 2)

            # ---- Duplicates (value-level, within this column) ----
            dup_pct = round(s.duplicated().mean() * 100, 2) if n else 0.0
            dup_score = round(100 - dup_pct, 2)

            # ---- Outliers / Data Types / Invalid ----
            if pd.api.types.is_numeric_dtype(s):
                if n and n_non_null:
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                    outlier_pct = round(out / n * 100, 2)
                else:
                    outlier_pct = 0.0
                outlier_score = round(100 - outlier_pct, 2)

                inf_count = int(np.isinf(non_null).sum()) if n_non_null else 0
                inf_pct = round(inf_count / n * 100, 2) if n else 0.0

                dtype_score = round(100 - inf_pct, 2)
                invalid_score = round(100 - inf_pct, 2)

            else:
                # Outliers aren't defined for non-numeric columns.
                outlier_score = 100.0

                text = non_null.astype(str)

                if n_non_null:
                    numeric_like = pd.to_numeric(text, errors="coerce")
                    convertible_frac = numeric_like.notna().mean()
                else:
                    convertible_frac = 0.0

                type_penalty_numeric_like = (
                    convertible_frac * 100 if convertible_frac > 0.8 else 0.0
                )

                if n_non_null:
                    types_seen = non_null.map(lambda v: type(v).__name__)
                    dominant_frac = types_seen.value_counts(normalize=True).iloc[0]
                    mixed_pct = round((1 - dominant_frac) * 100, 2)
                else:
                    mixed_pct = 0.0

                dtype_score = round(
                    max(0.0, 100 - max(type_penalty_numeric_like, mixed_pct)), 2
                )

                if n_non_null:
                    space_issue = int(text.str.strip().ne(text).sum())
                    empty_string = int((text == "").sum())
                    special_char = int(
                        text.str.contains(r"[^a-zA-Z0-9\s]", regex=True).sum()
                    )
                    invalid_pct = round(
                        min(
                            (space_issue + empty_string + special_char) / n * 100,
                            100.0,
                        ),
                        2,
                    )
                else:
                    invalid_pct = 0.0
                invalid_score = round(100 - invalid_pct, 2)

            quality_score = round(
                (
                    missing_score * w["missing"]
                    + dup_score * w["duplicates"]
                    + outlier_score * w["outliers"]
                    + dtype_score * w["dtypes"]
                    + invalid_score * w["invalid"]
                )
                / 100,
                2,
            )

            rows.append(
                {
                    "Column": col,
                    "Dtype": str(s.dtype),
                    "Missing Score": missing_score,
                    "Duplicate Score": dup_score,
                    "Outlier Score": outlier_score,
                    "Data Type Score": dtype_score,
                    "Invalid Score": invalid_score,
                    "Quality Score": quality_score,
                }
            )

        return pd.DataFrame(rows)

    # =====================================================
    # PII DETECTION
    # =====================================================

    def pii(
        self,
        min_confidence: float = _DEFAULT_PII_MIN_CONFIDENCE,
        sample: int | None = _DEFAULT_PII_SAMPLE,
        mask: bool = False,
        mask_char: str = "*",
    ):
        """
        Scan every column for values that look like personally
        identifiable information, using pattern matching plus a
        column-name hint (e.g. a column named "email" gets a
        confidence boost toward the EMAIL type).

        Detected types
        ---------------
        EMAIL, PHONE, POSSIBLE_ID (Aadhaar-like 12-digit numbers),
        POSSIBLE_PAN (Indian PAN format), IP, POSSIBLE_NAME,
        POSSIBLE_ADDRESS.

        The "POSSIBLE_" prefix on ID/PAN/NAME/ADDRESS is a reminder
        that those are pattern/heuristic guesses, not verified PII —
        always review before acting on them (e.g. before dropping or
        publishing columns).

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
            ``mask=False``; otherwise a masked copy of the underlying
            DataFrame.

        Examples
        --------
        >>> df.profile.pii()
        >>> df.profile.pii(mask=True)
        """

        if not (0 <= min_confidence <= 100):
            raise ValueError("min_confidence must be between 0 and 100.")

        detections: dict[str, tuple[str, float]] = {}

        for col in self.df.columns:
            result = self._detect_pii_column(col, self.df[col], sample)
            if result is None:
                continue
            pii_type, confidence = result
            if confidence >= min_confidence:
                detections[col] = (pii_type, confidence)

        if not mask:
            rows = [
                {"Column": col, "PII Type": pii_type, "Confidence %": confidence}
                for col, (pii_type, confidence) in detections.items()
            ]
            rows.sort(key=lambda r: r["Confidence %"], reverse=True)
            return pd.DataFrame(rows, columns=["Column", "PII Type", "Confidence %"])

        masked_df = self.df.copy()
        for col, (pii_type, _confidence) in detections.items():
            masked_df[col] = masked_df[col].map(
                lambda v: self._mask_value(v, pii_type, mask_char)
                if pd.notna(v)
                else v
            )
        return masked_df

    @staticmethod
    def _detect_pii_column(
        col_name, s: pd.Series, sample: int | None
    ) -> tuple[str, float] | None:
        non_null = s.dropna()
        if len(non_null) == 0:
            return None

        if sample and len(non_null) > sample:
            non_null = non_null.sample(n=sample, random_state=42)

        text = non_null.astype(str).str.strip()
        text = text[text != ""]
        n = len(text)
        if n == 0:
            return None

        name_lower = str(col_name).lower()

        candidates = [
            ("EMAIL", text.str.fullmatch(_EMAIL_RE).sum() / n * 100),
            ("PHONE", text.apply(_is_phone).sum() / n * 100),
            ("POSSIBLE_ID", text.str.fullmatch(_AADHAAR_RE).sum() / n * 100),
            ("POSSIBLE_PAN", text.str.fullmatch(_PAN_RE).sum() / n * 100),
            ("IP", text.apply(_is_ip).sum() / n * 100),
            ("POSSIBLE_NAME", text.apply(_looks_like_name).sum() / n * 100),
            ("POSSIBLE_ADDRESS", text.apply(_looks_like_address).sum() / n * 100),
        ]

        boosted = []
        for pii_type, confidence in candidates:
            hints = _PII_NAME_HINTS.get(pii_type, ())
            if any(h in name_lower for h in hints):
                confidence = min(confidence + _PII_NAME_HINT_BONUS, 100.0)
            boosted.append((pii_type, confidence))

        best_type, best_confidence = max(boosted, key=lambda t: t[1])
        return best_type, round(best_confidence, 2)

    @staticmethod
    def _mask_value(value, pii_type: str, mask_char: str) -> str:
        x = str(value)

        if pii_type == "EMAIL" and "@" in x:
            local, _, domain = x.partition("@")
            masked_local = local[0] + mask_char * max(len(local) - 1, 1)
            return f"{masked_local}@{domain}"

        if pii_type == "PHONE":
            digits = re.sub(r"\D", "", x)
            if len(digits) >= 4:
                return digits[:2] + mask_char * (len(digits) - 4) + digits[-2:]
            return mask_char * len(x)

        if pii_type == "POSSIBLE_ID":
            digits = re.sub(r"\D", "", x)
            if len(digits) >= 4:
                return mask_char * (len(digits) - 4) + digits[-4:]
            return mask_char * len(x)

        if pii_type == "POSSIBLE_PAN":
            if len(x) >= 3:
                return x[:2] + mask_char * (len(x) - 3) + x[-1]
            return mask_char * len(x)

        if pii_type == "IP":
            parts = x.split(".")
            if len(parts) == 4:
                return ".".join(parts[:3] + [mask_char * 3])
            return mask_char * len(x)

        if pii_type == "POSSIBLE_NAME":
            words = x.split()
            return " ".join(
                w[0] + mask_char * max(len(w) - 1, 1) for w in words
            )

        if pii_type == "POSSIBLE_ADDRESS":
            return "[REDACTED ADDRESS]"

        return mask_char * len(x)

    # =====================================================
    # CORRELATIONS
    # =====================================================

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
            by absolute value), with an ``R²`` column
            (``Correlation ** 2``). ``R²`` is the exact
            variance-explained figure from a single-predictor linear
            regression when ``method="pearson"``; for
            ``"spearman"``/``"kendall"`` it's a rank-based analog, not
            a literal linear R².
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
        """

        method = method.lower()
        by = by.lower()

        if method not in _VALID_CORR_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of "
                f"{sorted(_VALID_CORR_METHODS)}."
            )
        if by not in _VALID_CORR_BY:
            raise ValueError(
                f"Invalid by='{by}'. Must be one of {sorted(_VALID_CORR_BY)}."
            )

        numeric_df = self.df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            raise ValueError(
                "Need at least two numeric columns to compute correlations."
            )

        corr_matrix = numeric_df.corr(method=method).round(4)

        if by == "matrix":
            return corr_matrix

        return self._correlation_pairs(corr_matrix, threshold)

    @staticmethod
    def _correlation_pairs(
        corr_matrix: pd.DataFrame, threshold: float | None
    ) -> pd.DataFrame:
        cols = list(corr_matrix.columns)
        pairs = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = corr_matrix.loc[c1, c2]
                if pd.isna(val):
                    continue
                pairs.append(
                    {
                        "Column A": c1,
                        "Column B": c2,
                        "Correlation": round(float(val), 4),
                        "R²": round(float(val) ** 2, 4),
                        "_abs": abs(val),
                    }
                )

        result = pd.DataFrame(
            pairs, columns=["Column A", "Column B", "Correlation", "R²", "_abs"]
        )

        if threshold is not None:
            result = result[result["_abs"] >= threshold]

        return (
            result.sort_values("_abs", ascending=False)
            .drop(columns="_abs")
            .reset_index(drop=True)
        )

    # =====================================================
    # ALERTS
    # =====================================================

    def alerts(
        self,
        column_type: str = "All",
        min_severity: str = "low",
    ) -> pd.DataFrame:
        """
        Scan the DataFrame and surface a single, severity-ranked list
        of data-quality and risk warnings — missing data, duplicate
        rows, outliers, constant/near-constant columns, skew, high
        cardinality, columns stored as the wrong dtype, messy text,
        highly correlated numeric pairs, and potential PII — instead
        of requiring a column-by-column read of every other report.

        This is a convenience layer over the other sections
        (``quality``, ``ml``, ``etl``, ``correlations``, ``pii``): it
        doesn't compute anything those don't already compute, it just
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

        min_severity = min_severity.lower()
        column_type = column_type.lower()

        if min_severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid min_severity '{min_severity}'. Must be one of "
                f"{sorted(_VALID_SEVERITIES)}."
            )
        if column_type not in _VALID_COLUMN_TYPES:
            raise ValueError(
                f"Invalid column_type '{column_type}'. Must be one of "
                f"{sorted(_VALID_COLUMN_TYPES)}."
            )

        if column_type == "numeric":
            df = self.df.select_dtypes(include="number")
        elif column_type == "categorical":
            df = self.df.select_dtypes(include=["object", "category", "string"])
        else:
            df = self.df

        if df.shape[1] == 0:
            raise ValueError(f"No columns match column_type='{column_type}'.")

        rows = []
        n_rows = len(df)

        # ---- table-level: duplicate rows ----
        if n_rows:
            dup_row_pct = round(df.duplicated().mean() * 100, 2)
            if dup_row_pct > 0:
                sev = (
                    "high"
                    if dup_row_pct > 20
                    else ("medium" if dup_row_pct > 5 else "low")
                )
                rows.append(
                    self._alert_row(
                        sev,
                        "(table)",
                        "Duplicate Rows",
                        f"{dup_row_pct}% of rows are exact duplicates.",
                    )
                )

        # ---- table-level: highly correlated numeric pairs ----
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] >= 2:
            corr_matrix = numeric_df.corr(method="pearson").round(4)
            pairs = self._correlation_pairs(
                corr_matrix, threshold=_ALERT_CORRELATION_THRESHOLD
            )
            for _, r in pairs.iterrows():
                sev = "high" if abs(r["Correlation"]) >= 0.95 else "medium"
                rows.append(
                    self._alert_row(
                        sev,
                        "(table)",
                        "Highly Correlated Pair",
                        f"{r['Column A']} & {r['Column B']} correlation = "
                        f"{r['Correlation']}.",
                    )
                )

        # ---- column-level checks ----
        for col in df.columns:
            s = df[col]
            n = len(s)
            if n == 0:
                continue

            missing_pct = round(s.isna().mean() * 100, 2)
            if missing_pct > 0:
                sev = (
                    "high"
                    if missing_pct > 50
                    else ("medium" if missing_pct > 10 else "low")
                )
                rows.append(
                    self._alert_row(
                        sev, col, "Missing Values", f"{missing_pct}% missing."
                    )
                )

            non_null = s.dropna()
            n_non_null = len(non_null)

            if pd.api.types.is_numeric_dtype(s):
                var = s.var() if n > 1 else 0.0

                if n > 1 and (pd.isna(var) or var == 0):
                    rows.append(
                        self._alert_row(
                            "high",
                            col,
                            "Constant Column",
                            "Column has zero variance (a single unique value).",
                        )
                    )
                else:
                    skew = s.skew()
                    if pd.notna(skew) and abs(skew) > 1:
                        sev = "medium" if abs(skew) > 2 else "low"
                        rows.append(
                            self._alert_row(
                                sev,
                                col,
                                "Skewed Distribution",
                                f"Skewness = {round(skew, 2)}.",
                            )
                        )

                    if n_non_null:
                        q1, q3 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q3 - q1
                        out_pct = round(
                            ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                            / n
                            * 100,
                            2,
                        )
                        if out_pct > 0:
                            sev = (
                                "high"
                                if out_pct > 25
                                else ("medium" if out_pct > 10 else "low")
                            )
                            rows.append(
                                self._alert_row(
                                    sev,
                                    col,
                                    "Outliers",
                                    f"{out_pct}% of values are IQR outliers.",
                                )
                            )

                        zero_pct = round((s == 0).mean() * 100, 2)
                        if zero_pct > 50:
                            rows.append(
                                self._alert_row(
                                    "medium",
                                    col,
                                    "Zero-Heavy Column",
                                    f"{zero_pct}% of values are zero.",
                                )
                            )

                unique_pct = round(s.nunique() / n * 100, 2) if n else 0.0
                if n > 10 and unique_pct > 95:
                    rows.append(
                        self._alert_row(
                            "low",
                            col,
                            "Possible ID Column",
                            f"{unique_pct}% unique values — may be an "
                            f"identifier, not a feature.",
                        )
                    )

            else:
                unique_count = s.nunique()
                unique_pct = round(unique_count / n * 100, 2) if n else 0.0

                if n > 10 and unique_pct > 95:
                    rows.append(
                        self._alert_row(
                            "low",
                            col,
                            "Possible ID Column",
                            f"{unique_pct}% unique values — may be an "
                            f"identifier, not a feature.",
                        )
                    )
                elif unique_count > 50 and unique_pct > 50:
                    rows.append(
                        self._alert_row(
                            "medium",
                            col,
                            "High Cardinality",
                            f"{unique_count} unique values — consider "
                            f"hashing or target encoding.",
                        )
                    )

                if n_non_null:
                    vc = non_null.value_counts(normalize=True)
                    if len(vc) and vc.iloc[0] > 0.95:
                        rows.append(
                            self._alert_row(
                                "medium",
                                col,
                                "Near-Constant Column",
                                f"'{vc.index[0]}' accounts for "
                                f"{round(vc.iloc[0] * 100, 2)}% of values.",
                            )
                        )

                    text = non_null.astype(str)

                    numeric_like = pd.to_numeric(text, errors="coerce")
                    convertible_frac = numeric_like.notna().mean()
                    if convertible_frac > 0.8:
                        rows.append(
                            self._alert_row(
                                "medium",
                                col,
                                "Wrong Data Type",
                                f"{round(convertible_frac * 100, 2)}% of "
                                f"values look numeric but the column is "
                                f"stored as {s.dtype}.",
                            )
                        )

                    empty_pct = round((text == "").mean() * 100, 2)
                    if empty_pct > 0:
                        sev = "medium" if empty_pct > 5 else "low"
                        rows.append(
                            self._alert_row(
                                sev,
                                col,
                                "Empty Strings",
                                f"{empty_pct}% empty string values.",
                            )
                        )

                    space_pct = round(
                        text.str.strip().ne(text).mean() * 100, 2
                    )
                    if space_pct > 0:
                        rows.append(
                            self._alert_row(
                                "low",
                                col,
                                "Extra Whitespace",
                                f"{space_pct}% of values have leading/"
                                f"trailing whitespace.",
                            )
                        )

        # ---- PII ----
        for col in df.columns:
            result = self._detect_pii_column(col, df[col], _DEFAULT_PII_SAMPLE)
            if result is None:
                continue
            pii_type, confidence = result
            if confidence >= _DEFAULT_PII_MIN_CONFIDENCE:
                sev = "high" if confidence >= 90 else "medium"
                rows.append(
                    self._alert_row(
                        sev,
                        col,
                        "Potential PII",
                        f"Detected as {pii_type} ({confidence}% confidence).",
                    )
                )

        result = pd.DataFrame(
            rows, columns=["Severity", "Column", "Alert", "Details", "_rank"]
        )
        if result.empty:
            return result.drop(columns="_rank")

        keep_rank = _SEVERITY_ORDER[min_severity]
        result = result[result["_rank"] <= keep_rank]
        result = (
            result.sort_values(["_rank", "Column"])
            .drop(columns="_rank")
            .reset_index(drop=True)
        )
        return result

    @staticmethod
    def _alert_row(sev: str, column, alert: str, details: str) -> dict:
        return {
            "Severity": sev.capitalize(),
            "Column": column,
            "Alert": alert,
            "Details": details,
            "_rank": _SEVERITY_ORDER[sev],
        }

    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    def _ml_info(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for col in df.columns:
            s = df[col]
            n = len(s)

            unique = s.nunique()
            unique_pct = (unique / n * 100) if n else 0.0

            row = {
                "Column": col,
                "Dtype": str(s.dtype),
                "Feature Type": None,
                "Unique Count": unique,
                "Unique %": round(unique_pct, 2),
                "Missing %": round(s.isna().mean() * 100, 2),
                "Variance": None,
                "Skewness": None,
                "Outlier %": None,
                "Cardinality": None,
                "Encoding Suggestion": None,
                "Scaling Suggestion": None,
                "Transformation": None,
                "Feature Status": "Good",
            }

            # Numeric
            if pd.api.types.is_numeric_dtype(s):
                row["Feature Type"] = "Numeric"
                row["Variance"] = round(s.var(), 4)

                skew = s.skew()
                row["Skewness"] = round(skew, 4)

                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1

                out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                row["Outlier %"] = round(out / n * 100, 2) if n else 0.0

                row["Scaling Suggestion"] = (
                    "RobustScaler" if row["Outlier %"] > 5 else "StandardScaler"
                )

                row["Transformation"] = (
                    "Log Transform" if abs(skew) > 1 else "None"
                )

                if s.var() == 0:
                    row["Feature Status"] = "Constant Feature"
                elif unique_pct > 95:
                    row["Feature Status"] = "Possible ID"

            # Categorical
            else:
                row["Feature Type"] = "Categorical"

                if unique <= 10:
                    row["Cardinality"] = "Low"
                    row["Encoding Suggestion"] = "OneHot Encoding"
                elif unique_pct < 50:
                    row["Cardinality"] = "Medium"
                    row["Encoding Suggestion"] = "Target Encoding"
                else:
                    row["Cardinality"] = "High"
                    row["Encoding Suggestion"] = "Hash Encoding"

                if unique_pct > 95:
                    row["Feature Status"] = "Possible ID"

            rows.append(row)

        return pd.DataFrame(rows)

    # =====================================================
    # ETL
    # =====================================================

    def _etl_info(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for col in df.columns:
            s = df[col]
            n = len(s)

            missing_count = int(s.isna().sum())
            missing_pct = round(s.isna().mean() * 100, 2)

            unique_count = int(s.nunique())
            duplicate_count = int(s.duplicated().sum())

            row = {
                "Column": col,
                "Dtype": str(s.dtype),
                "Rows": n,
                "Missing Count": missing_count,
                "Missing %": missing_pct,
                "Unique Count": unique_count,
                "Unique %": round(unique_count / n * 100, 2) if n else 0.0,
                "Duplicate Count": duplicate_count,
                "Duplicate %": round(duplicate_count / n * 100, 2) if n else 0.0,
                "Issue": "None",
                "Issue Count": 0,
                "Recommended Action": "No Action",
                "ETL Status": "Ready",
            }

            issues = []

            # Missing Check
            if missing_pct > 0:
                issues.append("Missing Values")
                row["Issue Count"] += missing_count
                row["Recommended Action"] = "Impute Values"

            # Duplicate Check
            if duplicate_count > 0:
                issues.append("Duplicate Values")
                row["Issue Count"] += duplicate_count
                row["Recommended Action"] = "Remove Duplicates"

            # Numeric Validation
            if pd.api.types.is_numeric_dtype(s):
                negative_count = int((s < 0).sum())
                zero_count = int((s == 0).sum())

                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1

                outlier_count = int(
                    ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                )

                row.update(
                    {
                        "Negative Count": negative_count,
                        "Zero Count": zero_count,
                        "Outlier Count": outlier_count,
                        "Outlier %": round(outlier_count / n * 100, 2)
                        if n
                        else 0.0,
                    }
                )

                if negative_count > 0:
                    issues.append("Negative Values")

                if outlier_count > 0:
                    issues.append("Outliers")

            # Text Validation
            else:
                text = s.dropna().astype(str)

                space_issue = int(text.str.strip().ne(text).sum())
                empty_string = int((text == "").sum())
                special_char = int(
                    text.str.contains(r"[^a-zA-Z0-9\s]", regex=True).sum()
                )

                row.update(
                    {
                        "Space Issue Count": space_issue,
                        "Empty String Count": empty_string,
                        "Special Character Count": special_char,
                    }
                )

                if space_issue > 0:
                    issues.append("Extra Spaces")

                if empty_string > 0:
                    issues.append("Empty String")

            # Final Status
            if issues:
                row["Issue"] = ", ".join(issues)
                row["ETL Status"] = "Needs Cleaning"

            rows.append(row)

        return pd.DataFrame(rows)
