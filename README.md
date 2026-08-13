# drishtipy

### Lightweight Data Profiling & Data Quality for pandas

[![PyPI](https://img.shields.io/pypi/v/drishtipy.svg)](https://pypi.org/project/drishtipy/)
[![Python](https://img.shields.io/pypi/pyversions/drishtipy.svg)](https://pypi.org/project/drishtipy/)
[![Downloads](https://img.shields.io/pypi/dm/drishtipy.svg)](https://pypi.org/project/drishtipy/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#development)
[![GitHub](https://img.shields.io/badge/GitHub-drishtipy-black.svg)](https://github.com/drishtipy/drishtipy)

> **Understand your DataFrame in seconds.**

`drishtipy` is a lightweight, dependency-minimal pandas profiling toolkit for **data quality, statistics, ML readiness, ETL analysis, PII detection, correlations, dataset comparison, and HTML reporting.**

The goal is simple:

> **Load your DataFrame → profile it → find problems → understand the data → improve it.**

---

## ⚡ See It in Action

```python
import pandas as pd
import drishtipy
or
from drishtipy import profile_of  # ✅ full autocomplete + type checking

df = pd.read_csv("data.csv")

# Complete profile
df.profile.info() or profile_of(df).info()

# Data structure
df.profile.schema()

# Statistical analysis
df.profile.statistics()

# Data quality analysis
df.profile.quality()

# ML readiness & suggestions
df.profile.ml()

# ETL readiness & cleaning issues
df.profile.etl()

# Overall data quality score
df.profile.quality_score()

# Data quality alerts
df.profile.alerts()

# Potential PII detection
df.profile.pii()

# Correlation analysis
df.profile.correlations(by="pairs")

# Generate a shareable HTML report
df.profile.html("report.html")

# Compares two DataFrames and generates an HTML before/after report
df.profile.compare_html(df2, path="compare.html")
```

That's it.
No complex configuration. No wrapper object required.

Just import `drishtipy` and use:

```python
df.profile
```

---

## 🎯 What Can drishtipy Tell You?

Give `drishtipy` a pandas DataFrame and quickly answer questions like:

- What columns does my dataset contain?
- Which columns have missing values?
- Where are the duplicates?
- Which numeric columns contain outliers?
- How good is the overall data quality?
- Which columns may contain PII?
- Is this dataset ready for machine learning?
- What problems could affect an ETL pipeline?
- Which numeric features are strongly correlated?
- What changed after cleaning my data?
- Can I generate a report to share with someone else?

---

## 🚀 One DataFrame. Multiple Insights.

```text
                         pandas DataFrame
                                │
                                ▼
                         ┌─────────────┐
                         │  drishtipy  │
                         └─────────────┘
                                │
          ┌─────────────┬───────┼────────┬──────────────┐
          ▼             ▼       ▼        ▼              ▼
       Schema       Statistics Quality   ML             ETL
          │             │       │        │              │
          │             │       ▼        ▼              ▼
          │             │   Quality    Readiness     Cleaning
          │             │    Score                   Issues
          │             │       │
          │             ▼       ▼
          │        Correlations Alerts
          │
          ├──────────────► PII Detection
          │
          ├──────────────► Before / After Comparison
          │
          └──────────────► HTML Report
```

---

## ⭐ Core Features

| Feature | Purpose |
|---|---|
| **Schema** | Understand columns, dtypes, missing values, uniqueness and memory |
| **Statistics** | Descriptive statistics, quantiles, skewness, kurtosis and more |
| **Quality** | Missing values, duplicates, zeros and outliers |
| **Quality Score** | Overall and per-column data quality scoring |
| **Alerts** | High, medium and low severity data-quality issues |
| **PII Detection** | Detect potentially sensitive and identifiable data |
| **Correlations** | Find relationships and multicollinearity candidates |
| **ML Readiness** | Analyze feature suitability for machine learning |
| **ETL Readiness** | Identify common cleaning and transformation problems |
| **Comparison** | Compare DataFrames before and after transformations |
| **HTML Reports** | Generate standalone, shareable reports |
| **Large CSV** | Profile large CSV files using chunking and sampling |

---

## 📦 Installation

```bash
pip install drishtipy
```

Then:

```python
import pandas as pd
import drishtipy

df = pd.read_csv("data.csv")

df.profile.quality()
```

### Development installation

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

---

## 🐼 Designed for pandas

`drishtipy` extends the familiar pandas workflow with a `.profile` accessor.

Instead of:

```python
DataProfiler(df).info_dataframe(
    section="quality"
)
```

you can simply write:

```python
df.profile.quality()
```

The profiling results remain pandas-friendly, so you can continue to use normal pandas operations:

```python
report = df.profile.quality()

report.sort_values(
    "Missing Count",
    ascending=False
)
```

Filter them:

```python
report[
    report["Missing Count"] > 0
]
```

Or export them:

```python
report.to_csv(
    "quality_report.csv",
    index=False
)
```

---

## 📋 Schema Profiling

Inspect the structure of your DataFrame:

```python
df.profile.schema()
```

The schema report includes one row per column with:

- Column name and pandas dtype
- Non-null count, missing count, missing %
- Unique count, unique %
- Memory usage (bytes)

---

## 📊 Statistical Profiling

Generate descriptive statistics:

```python
df.profile.statistics()
```

One row per column, with (for numeric columns) count, mean, median, std,
min/Q1/Q2/Q3/95th/99th percentile/max, range, IQR, skewness, and kurtosis —
plus mode and mode frequency for every column regardless of dtype.

---

## 🔍 Data Quality Profiling

Analyze common data-quality problems:

```python
df.profile.quality()
```

The quality report includes:

- Duplicate value count (per column)
- Missing value count
- Zero count (numeric columns)
- IQR-based outlier count and outlier % (numeric columns)

Example:

```text
Column    Duplicate Count    Missing Count    Zero Count    Outlier Count    Outlier %
------------------------------------------------------------------------------------
age                     0                1             0                0        0.00
salary                  0                0             0                1       25.00
city                    1                0          None             None        None
```

---

## ⭐ Quality Score

Get a quick, weighted assessment of your dataset across five dimensions —
Missing Values, Duplicates, Outliers, Data Types, and Invalid Values:

```python
df.profile.quality_score()
```

Example output (a `Metric` / `Score` DataFrame):

```text
                 Metric  Score
 Overall Quality Score   88.4
         Missing Values  92.0
             Duplicates 100.0
               Outliers  71.0
             Data Types  95.0
         Invalid Values  84.0
```

For a column-level breakdown (one row per column, plus a per-column
`Quality Score`):

```python
df.profile.quality_score(by="column")
```

Filter by column type:

```python
df.profile.quality_score(column_type="numeric")
```

Custom scoring weights (unspecified dimensions keep their default weight of
20; values are normalized automatically, so they don't need to sum to 100):

```python
df.profile.quality_score(
    weights={
        "missing": 40,
        "outliers": 30,
    }
)
```

> **Note:** `Duplicates` at the overall level measures fully duplicated
> *rows*. At the column level (`by="column"`), it measures repeated
> *values* within that column instead — the two aren't the same metric.

---

## 🚨 Data Quality Alerts

Find important data-quality problems automatically, ranked by severity, in
one table instead of reading every section separately:

```python
df.profile.alerts()
```

Returns a DataFrame with columns `Severity`, `Column`, `Alert`, and
`Details` (table-wide issues like duplicate rows or correlated pairs use
`"(table)"` as the column value). Detected alert types include: Missing
Values, Duplicate Rows, Outliers, Constant Column, Near-Constant Column,
Skewed Distribution, High Cardinality, Possible ID Column, Wrong Data Type,
Empty Strings, Extra Whitespace, Zero-Heavy Column, Highly Correlated Pair,
and Potential PII.

Filter for critical issues only:

```python
df.profile.alerts(min_severity="high")
```

`min_severity` accepts `"high"` (High only), `"medium"` (High + Medium), or
`"low"` (everything — the default).

---

## 🔐 PII Detection

Detect potentially personally identifiable information:

```python
df.profile.pii()
```

Returns a DataFrame — `Column`, `PII Type`, `Confidence %` — sorted by
confidence, descending. Detected types:

| Type | Meaning |
|---|---|
| `EMAIL` | Email addresses |
| `PHONE` | Phone numbers (Indian mobile pattern + general) |
| `POSSIBLE_ID` | Aadhaar-shaped 12-digit numbers |
| `POSSIBLE_PAN` | Indian PAN format (`ABCDE1234F`) |
| `IP` | IPv4 addresses |
| `POSSIBLE_NAME` | Free-text columns that look like personal names |
| `POSSIBLE_ADDRESS` | Free-text columns that look like postal addresses |

The `POSSIBLE_` prefix is a reminder that ID/PAN/Name/Address detections are
pattern/heuristic guesses, not verified PII — always review before acting on
them (e.g. before dropping or publishing columns). `EMAIL`, `PHONE`, and
`IP` use stricter pattern matching.

Filter by confidence:

```python
df.profile.pii(min_confidence=80)
```

For large DataFrames, only the first `sample` non-null values per column
are scanned by default (for speed):

```python
df.profile.pii(sample=2000)   # default
df.profile.pii(sample=None)   # scan every value instead
```

Generate a masked copy instead of a report:

```python
masked_df = df.profile.pii(mask=True)
```

`mask=True` returns a masked **copy** of the DataFrame — the original is
never modified. Masking is type-aware (emails keep the domain, phone/ID
numbers keep the last few digits, addresses are fully redacted, etc.).

---

## 🧮 Correlation Analysis

Analyze relationships between numerical columns (Pearson by default):

```python
df.profile.correlations()
```

Returns the full column-by-column correlation matrix. Get a tidy,
one-row-per-pair view instead, sorted by strength:

```python
df.profile.correlations(by="pairs")
```

Find only strongly correlated pairs — useful for spotting multicollinearity
candidates before modeling:

```python
df.profile.correlations(by="pairs", threshold=0.8)
```

Supported methods:

```python
df.profile.correlations(method="pearson")
df.profile.correlations(method="spearman")
df.profile.correlations(method="kendall")
```

Raises `ValueError` if fewer than two numeric columns are available.

---

## 🤖 ML Readiness

Analyze whether your DataFrame is suitable for machine-learning workflows:

```python
df.profile.ml()
```

One row per column, including: feature type, unique count/%, missing %,
variance, skewness, outlier %, cardinality, encoding suggestion, scaling
suggestion, recommended transformation, and an overall feature status
(e.g. `"Good"`) flagging columns that may need attention before modeling.

---

## 🔄 ETL Readiness

Analyze your DataFrame for common ETL and cleaning problems:

```python
df.profile.etl()
```

One row per column, including: missing/duplicate/unique counts and %,
zero count, negative count, empty-string count, whitespace-issue count,
special-character count, outlier count/%, an `Issue`/`Issue Count` summary,
an `ETL Status` (e.g. `"Ready"`), and a `Recommended Action`.

---

## 🔬 Before / After DataFrame Comparison

`drishtipy` includes `DataComparator` for comparing DataFrames before and after transformations.

Useful for:

- ETL pipelines
- Data cleaning
- Data transformation
- Feature engineering
- Data validation
- Preprocessing workflows

```python
from drishtipy import DataComparator

comparator = DataComparator()

comparison = comparator.compare_dataframe(
    before_df,
    after_df
)
```

Or through the pandas accessor:

```python
df.profile.compare(df2)
```

### Comparison levels

Supported levels:

```text
All
Dataset
Schema
Quality
ML
```

(`"quality"` and `"ml"` currently produce the same column-level report.)

Example:

```python
comparator.compare_dataframe(
    before_df,
    after_df,
    level="quality"
)
```

The result is a long-format `DataFrame` with columns `Section`, `Metric`,
`Before`, `After`, `Change`, and `Status` (not every row populates every
column). Raises `TypeError` if either argument isn't a DataFrame, and
`ValueError` for an unrecognized `level`.

### Changes only

`compare_dataframe` can return dozens of rows on wide DataFrames. Keep only
the rows where something actually changed:

```python
comparator.compare_dataframe(
    before_df,
    after_df,
    changes_only=True
)
```

### Compact summary

A one-row-per-column view instead of the long-format table:

```python
comparator.summary(
    before_df,
    after_df
)
```

```text
   Column     Status                          Summary
0     age    Changed  missing 1->0; outliers 1->0; mean 122.5->35.17
1    city  Unchanged                         No change
2 new_col      Added                     Column added
3 old_col    Removed                   Column removed
```

### Specific columns

```python
comparator.compare_dataframe(
    before_df,
    after_df,
    level="quality",
    columns=["age", "salary"]
)
```

A column missing from one side is still reported as added/removed. Raises
`ValueError` if a name in `columns` isn't present in either DataFrame. Note:
`level="dataset"` metrics (row/column counts, memory, etc.) always reflect
the full `before`/`after` DataFrames, not the selection.

---

## 🌐 HTML Reports

Generate a standalone, dark-themed HTML report instead of raw DataFrames —
handy for sharing a report without a notebook:

```python
df.profile.html(
    "profile_report.html"
)
```

Comparison reports:

```python
df.profile.compare_html(
    df2,
    path="comparison_report.html"
)
```

Explicit classes also support HTML output:

```python
DataProfiler(df).to_html(
    path="profile_report.html"
)

DataComparator().to_html(
    before_df,
    after_df,
    path="comparison_report.html"
)
```

Both accept the same filtering arguments as their DataFrame-returning
counterparts (`section`/`column_type` for `DataProfiler`, `level`/
`changes_only`/`columns` for `DataComparator`), plus `title` for a custom
page heading. Omitting `path` returns the HTML as a string instead of
writing a file.

---

## 📁 Large CSV Profiling

Profile large CSV files without loading the entire source into memory:

```python
from drishtipy import DataProfiler

profiler = DataProfiler.from_csv(
    "huge_dataset.csv",
    sample_size=100_000,
    chunksize=50_000,
    random_state=42
)
```

Then:

```python
profiler.info_dataframe()
```

Check whether sampling was used:

```python
profiler.is_sampled
```

Get the exact source row count:

```python
profiler.total_rows_in_source
```

### How sampling works

```text
CSV file
   │
   ├── Pass 1
   │     └── Count total rows
   │
   ├── Pass 2
   │     └── Select a random sample
   │
   └── Profile the sample
```

The file is streamed in chunks twice — once to count exact total rows, once
to keep each row with probability `sample_size / total_rows` (a random,
roughly-uniform sample). If the file has fewer rows than `sample_size`,
it's loaded in full and `is_sampled` stays `False`.

When sampling is used, the source row count (`total_rows_in_source`) is
exact, while statistics/ML sections computed from the sample are estimates
of the complete dataset. `to_html()` on a sampled profiler automatically
notes the sample size vs. total rows in the report.

Additional keyword arguments can be forwarded to `pandas.read_csv()`:

```python
DataProfiler.from_csv(
    "data.csv",
    usecols=["age", "salary"],
    dtype={"age": "float64"},
    sep=",",
    parse_dates=["date"]
)
```

---

## 🧱 Explicit Class API

The pandas accessor is recommended, but explicit classes are also available
— useful if you want to keep the profiler object around for repeated calls,
or prefer not to rely on accessor registration:

```python
import pandas as pd

from drishtipy import DataProfiler

df = pd.DataFrame({
    "age": [25, 32, None, 47],
    "city": ["Delhi", "Mumbai", "Delhi", "Pune"]
})

profiler = DataProfiler(df)

profiler.info_dataframe(
    section="schema"
)

profiler.quality_score()
profiler.alerts()
profiler.pii()
profiler.correlations()
```

Raises `TypeError` if `df` isn't a `pandas.DataFrame`.

---

## ⚙️ `info_dataframe()`

The underlying profiler API provides:

```python
profiler.info_dataframe(
    section="All",
    column_type="All"
)
```

### `section`

Supported values (case-insensitive):

```text
All
Schema
Statistics
Quality
ML
ETL
```

`"All"` returns a `dict` mapping section name -> `DataFrame`:

```python
{
    "Schema": ...,
    "Statistics": ...,
    "Quality": ...,
    "ML": ...,
    "ETL": ...
}
```

Any other value returns just that section's `DataFrame`.

### `column_type`

Supported values (case-insensitive):

```text
All
Numeric
Categorical
```

Example:

```python
profiler.info_dataframe(
    section="statistics",
    column_type="numeric"
)
```

Raises `ValueError` if `section` or `column_type` isn't recognized, or if
`column_type` filters out every column.

---

## 🧠 Design Philosophy

### pandas-first

`drishtipy` extends pandas rather than replacing it.

### Lightweight

The core package intentionally keeps dependencies minimal — pandas is the
only required dependency.

### Non-destructive

Profiling operations analyze the DataFrame without modifying it. The one
exception is explicit: `df.profile.pii(mask=True)` returns a masked
**copy**, never mutating the original.

### Composable

Reports remain compatible with normal pandas operations — sort, filter,
export, or display them however you like.

### Human-readable

HTML reports and summaries are designed to make data problems easy to
understand at a glance.

---

## 🧪 Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Recommended workflow:

```bash
git clone <repository>
cd drishtipy

pip install -e ".[dev]"

pytest
```

---

## 🗺️ Roadmap

Potential future capabilities include:

- Advanced validation rules
- Automatic cleaning suggestions
- Smart semantic data-type/data-dictionary detection
- Time-series profiling
- Advanced ML-readiness checks
- Data drift detection
- Statistical distribution analysis
- Excel / JSON / Parquet / SQL database profiling
- Expanded interactive HTML dashboards
- Additional data-quality checks

---

## 📦 Package Information

| Property | Value |
|---|---|
| Package | `drishtipy` |
| Import | `drishtipy` |
| Current Version | `0.5.2` |
| Python | `>=3.8` |
| License | MIT |
| Primary Dependency | pandas (>=1.3) |

Install:

```bash
pip install drishtipy
```

Import:

```python
import drishtipy
```

Use:

```python
df.profile.quality()
```

---

## 📄 License

`drishtipy` is released under the MIT License.
