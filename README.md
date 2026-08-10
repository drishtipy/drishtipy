# drishtipy

### Lightweight Data Profiling & Data Quality for pandas

[![PyPI](https://img.shields.io/pypi/v/drishtipy.svg)](https://pypi.org/project/drishtipy/)
[![Python](https://img.shields.io/pypi/pyversions/drishtipy.svg)](https://pypi.org/project/drishtipy/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#development)

> **Understand your DataFrame in seconds.**

`drishtipy` is a lightweight, dependency-minimal pandas profiling toolkit for **data quality, statistics, ML readiness, ETL analysis, PII detection, correlations, dataset comparison, and HTML reporting.**

The goal is simple:

> **Load your DataFrame → profile it → find problems → understand the data → improve it.**

---

## ⚡ See It in Action

```python
import pandas as pd 
import drishtipy 
df = pd.read_csv("data.csv") 

# Complete profile 
df.profile.info() 

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

#Compares two DataFrames and generates an HTML before/after report
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

The schema report provides information such as:

- Column name
- pandas dtype
- Non-null count
- Missing count
- Missing percentage
- Unique count
- Uniqueness
- Memory usage

---

## 📊 Statistical Profiling

Generate descriptive statistics:

```python
df.profile.statistics()
```

Depending on the data type, the report can include:

- Count
- Mean
- Median
- Minimum
- Maximum
- Standard deviation
- Quantiles
- Mode
- Skewness
- Kurtosis
- Cardinality

---

## 🔍 Data Quality Profiling

Analyze common data-quality problems:

```python
df.profile.quality()
```

The quality report includes:

- Missing values
- Duplicate values
- Zero values
- IQR-based outliers
- Outlier percentage

Example:

```text
Column    Missing Count    Duplicate Count    Outlier Count    Outlier %
-----------------------------------------------------------------------
age       1                0                  0                0.00
salary    0                0                  1                25.00
city      0                1                  -                -
```

---

## ⭐ Quality Score

Get a quick assessment of your dataset:

```python
df.profile.quality_score()
```

Example:

```text
Overall Quality Score: 91.4 / 100
Grade: A

Missing Values    96.2
Duplicates       100.0
Outliers           88.7
Consistency        92.5
Completeness       95.4
```

For a column-level breakdown:

```python
df.profile.quality_score(
    by="column"
)
```

Filter by column type:

```python
df.profile.quality_score(
    column_type="numeric"
)
```

Custom scoring weights:

```python
df.profile.quality_score(
    weights={
        "missing": 40,
        "outliers": 30
    }
)
```

> **Note:** The exact score dimensions, ranges, and weight validation should match the implementation shipped in the package.

---

## 🚨 Data Quality Alerts

Find important data-quality problems automatically:

```python
df.profile.alerts()
```

Filter for critical issues:

```python
df.profile.alerts(
    min_severity="high"
)
```

Alerts are organized by severity:

```text
High
Medium
Low
```

---

## 🔐 PII Detection

Detect potentially personally identifiable information:

```python
df.profile.pii()
```

Potential detections can include:

- Email addresses
- Phone numbers
- Identification numbers
- IP addresses
- Other recognizable sensitive patterns

Filter by confidence:

```python
df.profile.pii(
    min_confidence=80
)
```

For large DataFrames, scan a sample:

```python
df.profile.pii(
    sample=2000
)
```

Scan the complete DataFrame:

```python
df.profile.pii(
    sample=None
)
```

Generate a masked copy:

```python
masked_df = df.profile.pii(
    mask=True
)
```

The original DataFrame is not modified.

---

## 🧮 Correlation Analysis

Analyze relationships between numerical columns:

```python
df.profile.correlations()
```

Get tidy correlation pairs:

```python
df.profile.correlations(
    by="pairs"
)
```

Find strong relationships:

```python
df.profile.correlations(
    by="pairs",
    threshold=0.8
)
```

Supported methods:

```python
df.profile.correlations(method="pearson")
df.profile.correlations(method="spearman")
df.profile.correlations(method="kendall")
```

---

## 🤖 ML Readiness

Analyze whether your DataFrame is suitable for machine-learning workflows:

```python
df.profile.ml()
```

The ML report can identify:

- Numeric features
- Categorical features
- High-cardinality columns
- Potential identifiers
- Encoding recommendations
- Scaling recommendations
- Feature status
- Potentially problematic columns

If implemented by the current version, a high-level readiness summary can be exposed through:

```python
df.profile.ml_readiness()
```

---

## 🔄 ETL Readiness

Analyze your DataFrame for common ETL and cleaning problems:

```python
df.profile.etl()
```

The ETL report can identify:

- Missing values
- Duplicate values
- Negative values
- Outliers
- Empty strings
- Text-quality problems
- Potential cleaning actions

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

Example:

```python
comparator.compare_dataframe(
    before_df,
    after_df,
    level="quality"
)
```

### Changes only

```python
comparator.compare_dataframe(
    before_df,
    after_df,
    changes_only=True
)
```

### Compact summary

```python
comparator.summary(
    before_df,
    after_df
)
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

---

## 🌐 HTML Reports

Generate a standalone HTML report:

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

If supported by the implementation, omitting `path` returns the HTML as a string.

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

When sampling is used, the source row count is exact, while statistics calculated from the sample are estimates of the complete dataset.

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

The pandas accessor is recommended, but explicit classes are also available:

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
```

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

Supported values:

```text
All
Schema
Statistics
Quality
ML
ETL
```

Values are case-insensitive.

`All` returns a dictionary containing the five core profiling sections:

```python
{
    "schema": ...,
    "statistics": ...,
    "quality": ...,
    "ml": ...,
    "etl": ...
}
```

A specific section returns that section's DataFrame.

### `column_type`

Supported values:

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

---

## 🧠 Design Philosophy

### pandas-first

`drishtipy` extends pandas rather than replacing it.

### Lightweight

The core package intentionally keeps dependencies minimal.

### Non-destructive

Profiling operations analyze the DataFrame without modifying it.

### Composable

Reports remain compatible with normal pandas operations.

### Human-readable

HTML reports and summaries are designed to make data problems easy to understand.

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
- Smart semantic data-type detection
- Data dictionary generation
- Time-series profiling
- Advanced ML-readiness checks
- Data drift detection
- Statistical distribution analysis
- Excel profiling
- JSON profiling
- Parquet profiling
- SQL database profiling
- Expanded interactive HTML dashboards
- Additional data-quality checks

---

## 📦 Package Information

| Property | Value |
|---|---|
| Package | `drishtipy` |
| Import | `drishtipy` |
| Current Version | `0.4.0` |
| Python | `>=3.8` |
| License | MIT |
| Primary Dependency | pandas |

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
