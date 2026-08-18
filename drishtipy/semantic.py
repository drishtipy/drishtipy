"""
Semantic column type detection.

Pandas dtypes tell you *storage* type (int64, object, ...), not
*meaning* (is this an identifier? a percentage? a phone number?).
This module infers the latter using a mix of dtype, cardinality,
column-name hints, and value-pattern checks — used by the
Relationship Discovery Engine to decide which statistical method
applies to a given pair of columns, and useful on its own.

This is intentionally a set of pragmatic heuristics, not a certified
classifier — see ``detect_semantic_type``'s docstring for exactly
what each label means and how confident to be in it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .core import _EMAIL_RE, _AADHAAR_RE, _is_phone, _is_ip

SEMANTIC_TYPES = (
    "id",
    "boolean",
    "datetime",
    "date",
    "email",
    "phone",
    "latitude",
    "longitude",
    "currency",
    "percentage",
    "numeric",
    "categorical",
    "text",
    "constant",
)

_ID_NAME_HINTS = ("id", "_id", "code", "uuid", "guid", "key")
_LAT_NAME_HINTS = ("lat", "latitude")
_LON_NAME_HINTS = ("lon", "lng", "longitude")
_CURRENCY_NAME_HINTS = (
    "price", "salary", "cost", "amount", "revenue", "income",
    "fee", "wage", "expense", "value", "payment", "budget",
)
_PERCENTAGE_NAME_HINTS = ("pct", "percent", "percentage", "rate", "ratio")

_BOOL_TRUE_STRINGS = {"true", "yes", "y", "1"}
_BOOL_FALSE_STRINGS = {"false", "no", "n", "0"}

_ID_UNIQUENESS_THRESHOLD = 0.95
_TEXT_UNIQUENESS_THRESHOLD = 0.5
_TEXT_MIN_AVG_LENGTH = 25
_DATETIME_PARSE_THRESHOLD = 0.9
_SAMPLE_SIZE = 500


def _name_hints(name: str, hints: tuple[str, ...]) -> bool:
    lower = str(name).lower()
    return any(h in lower for h in hints)


def _is_boolean_like(non_null: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(non_null):
        return True
    uniques = set(str(v).strip().lower() for v in non_null.unique())
    if len(uniques) > 2:
        return False
    if not uniques:
        return False
    return uniques <= (_BOOL_TRUE_STRINGS | _BOOL_FALSE_STRINGS)


def _datetime_parse_fraction(sample: pd.Series) -> float:
    if len(sample) == 0:
        return 0.0
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
    except (ValueError, TypeError):
        return 0.0
    return float(parsed.notna().mean())


def detect_semantic_type(s: pd.Series, name: str | None = None) -> str:
    """
    Infer the semantic type of a pandas Series.

    Returns one of: ``"id"``, ``"boolean"``, ``"datetime"``,
    ``"date"``, ``"email"``, ``"phone"``, ``"latitude"``,
    ``"longitude"``, ``"currency"``, ``"percentage"``, ``"numeric"``,
    ``"categorical"``, ``"text"``, ``"constant"``.

    This uses dtype, cardinality/uniqueness ratio, column-name
    patterns, and value-pattern checks (email/phone regex, date
    parsing) — it is a heuristic classifier, not a guarantee. Column
    names are only ever used as a *tiebreaker or confidence boost*,
    never as the sole signal, except for latitude/longitude and
    currency/percentage, which are inherently ambiguous from values
    alone (a plain float column has no way to say "I'm a percentage"
    without either a name hint or a 0-1/0-100 range convention).

    Parameters
    ----------
    s : pandas.Series
    name : str, optional
        Column name, used for the name-hint signals above. Defaults
        to ``s.name`` if not given.

    Returns
    -------
    str
    """
    name = name if name is not None else (s.name or "")
    non_null = s.dropna()
    n = len(non_null)

    if n == 0:
        return "categorical"

    unique_count = non_null.nunique()
    unique_ratio = unique_count / n

    if unique_count == 1:
        return "constant"

    if _is_boolean_like(non_null):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_numeric_dtype(s):
        if _name_hints(name, _ID_NAME_HINTS) and unique_ratio >= _ID_UNIQUENESS_THRESHOLD:
            return "id"
        if _name_hints(name, _LAT_NAME_HINTS) and non_null.between(-90, 90).mean() > 0.99:
            return "latitude"
        if _name_hints(name, _LON_NAME_HINTS) and non_null.between(-180, 180).mean() > 0.99:
            return "longitude"
        if _name_hints(name, _PERCENTAGE_NAME_HINTS):
            return "percentage"
        if _name_hints(name, _CURRENCY_NAME_HINTS):
            return "currency"
        return "numeric"

    # Remaining: object / string / category dtype.
    sample = non_null if n <= _SAMPLE_SIZE else non_null.sample(
        n=_SAMPLE_SIZE, random_state=42
    )
    text = sample.astype(str).str.strip()
    text = text[text != ""]
    if len(text) == 0:
        return "categorical"

    if text.apply(lambda x: bool(_EMAIL_RE.fullmatch(x))).mean() > 0.9:
        return "email"
    if text.apply(_is_phone).mean() > 0.9:
        return "phone"

    date_frac = _datetime_parse_fraction(sample)
    if date_frac >= _DATETIME_PARSE_THRESHOLD:
        return "date"

    if unique_ratio >= _ID_UNIQUENESS_THRESHOLD and _name_hints(name, _ID_NAME_HINTS):
        return "id"

    avg_len = text.str.len().mean()
    if unique_ratio > _TEXT_UNIQUENESS_THRESHOLD and avg_len > _TEXT_MIN_AVG_LENGTH:
        return "text"

    if unique_ratio >= _ID_UNIQUENESS_THRESHOLD:
        return "id"

    return "categorical"


def semantic_type_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count how many columns fall into each semantic type."""
    counts: dict[str, int] = {}
    for col in df.columns:
        t = detect_semantic_type(df[col], col)
        counts[t] = counts.get(t, 0) + 1
    return counts
