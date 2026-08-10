import numpy as np
import pandas as pd
import pytest

from drishtipy import DataComparator


@pytest.fixture
def before_after():
    before = pd.DataFrame(
        {
            "age": [25, 32, 47, np.nan, 29, 400],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Mumbai"],
            "old_col": [1, 2, 3, 4, 5, 6],
        }
    )

    after = pd.DataFrame(
        {
            "age": [25, 32, 47, 30, 29, 48],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Mumbai"],
            "new_col": [1, 2, 3, 4, 5, 6],
        }
    )

    return before, after


def test_requires_dataframes(before_after):
    before, after = before_after
    with pytest.raises(TypeError):
        DataComparator().compare_dataframe([1, 2, 3], after)
    with pytest.raises(TypeError):
        DataComparator().compare_dataframe(before, [1, 2, 3])


def test_invalid_level(before_after):
    before, after = before_after
    with pytest.raises(ValueError):
        DataComparator().compare_dataframe(before, after, level="bogus")


def test_dataset_level_rows_and_missing(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="dataset")

    rows = report[report["Metric"] == "Rows"].iloc[0]
    assert rows["Before"] == 6 and rows["After"] == 6 and rows["Status"] == "Same"

    missing = report[report["Metric"] == "Missing Cells"].iloc[0]
    assert missing["Before"] == 1
    assert missing["After"] == 0
    assert missing["Status"] == "Improved"


def test_schema_level_detects_add_remove_and_common(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="schema")

    statuses = dict(zip(report["Metric"], report["Status"]))
    assert statuses["old_col"] == "Deleted Column"
    assert statuses["new_col"] == "New Column"
    # 'age' loses its NaN in `after`, so float64 -> int64 is a real dtype change
    assert statuses["age"] == "Dtype Changed"
    # 'city' dtype is unchanged -> no row emitted for it
    assert "city" not in statuses


def test_schema_level_detects_dtype_change():
    before = pd.DataFrame({"x": [1, 2, 3]})
    after = pd.DataFrame({"x": ["1", "2", "3"]})

    report = DataComparator().compare_dataframe(before, after, level="schema")
    row = report[report["Metric"] == "x"].iloc[0]
    assert row["Status"] == "Dtype Changed"


def test_quality_level_missing_fixed(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="quality")

    age_missing = report[
        (report["Section"] == "Missing") & (report["Metric"] == "age")
    ].iloc[0]
    assert age_missing["Before"] == 1
    assert age_missing["After"] == 0
    assert age_missing["Status"] == "Fixed"


def test_quality_level_outlier_reduced(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="quality")

    age_outlier = report[
        (report["Section"] == "Outlier") & (report["Metric"] == "age")
    ].iloc[0]
    # before has 400 as an extreme outlier; after replaces it with 48
    assert age_outlier["Before"] >= 1
    assert age_outlier["Status"] in ("Reduced", "Same")


def test_category_section_for_non_numeric(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="ml")

    city_row = report[
        (report["Section"] == "Category") & (report["Metric"] == "city")
    ].iloc[0]
    assert city_row["Before"] == city_row["After"] == 3
    assert city_row["Change"] == 0


def test_statistics_and_quantile_sections_present(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="ml")

    metrics = set(report[report["Section"] == "Statistics"]["Metric"])
    assert "age_mean" in metrics
    assert "age_std" in metrics

    quantile_metrics = set(report[report["Section"] == "Quantile"]["Metric"])
    assert "age_Q25" in quantile_metrics
    assert "age_Q95" in quantile_metrics


def test_all_level_includes_every_section(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(before, after, level="all")

    sections = set(report["Section"])
    assert {
        "Dataset",
        "Schema",
        "Missing",
        "Zero",
        "Statistics",
        "Quantile",
        "Outlier",
        "Category",
    }.issubset(sections)


def test_case_insensitive_level(before_after):
    before, after = before_after
    a = DataComparator().compare_dataframe(before, after, level="DATASET")
    b = DataComparator().compare_dataframe(before, after, level="dataset")
    pd.testing.assert_frame_equal(a, b)


def test_identical_dataframes_are_all_same():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    report = DataComparator().compare_dataframe(df, df.copy(), level="all")

    # No schema changes expected between identical frames
    assert report[report["Section"] == "Schema"].empty


def test_changes_only_drops_unchanged_rows(before_after):
    before, after = before_after
    full = DataComparator().compare_dataframe(before, after, level="all")
    filtered = DataComparator().compare_dataframe(
        before, after, level="all", changes_only=True
    )

    assert len(filtered) < len(full)
    # city is identical in before/after -> its rows should be gone
    assert not (
        (filtered["Section"].isin(["Missing", "Category"]))
        & (filtered["Metric"] == "city")
    ).any()
    # schema rows (added/removed columns) must survive the filter
    assert {"old_col", "new_col"}.issubset(set(filtered["Metric"]))


def test_changes_only_on_identical_dataframes_is_mostly_empty():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    filtered = DataComparator().compare_dataframe(
        df, df.copy(), level="all", changes_only=True
    )
    assert filtered.empty


def test_summary_requires_dataframes(before_after):
    before, after = before_after
    with pytest.raises(TypeError):
        DataComparator().summary([1, 2, 3], after)


def test_summary_one_row_per_column(before_after):
    before, after = before_after
    summary = DataComparator().summary(before, after)

    # old_col removed, new_col added, age + city common -> 4 rows total
    assert len(summary) == 4
    statuses = dict(zip(summary["Column"], summary["Status"]))
    assert statuses["old_col"] == "Removed"
    assert statuses["new_col"] == "Added"
    assert statuses["age"] == "Changed"
    assert statuses["city"] == "Unchanged"


def test_summary_notes_mention_missing_fix(before_after):
    before, after = before_after
    summary = DataComparator().summary(before, after)
    age_row = summary[summary["Column"] == "age"].iloc[0]
    assert "missing" in age_row["Summary"]


def test_compare_dataframe_columns_restricts_output(before_after):
    before, after = before_after
    # level="ml" triggers only column-level rows (no dataset-level rows mixed in)
    report = DataComparator().compare_dataframe(
        before, after, level="ml", columns=["age"]
    )
    assert set(report["Metric"].str.split("_").str[0]) <= {"age"}
    assert "city" not in report["Metric"].values


def test_compare_dataframe_columns_unknown_raises(before_after):
    before, after = before_after
    with pytest.raises(ValueError):
        DataComparator().compare_dataframe(before, after, columns=["nonexistent"])


def test_compare_dataframe_columns_handles_added_removed(before_after):
    before, after = before_after
    # old_col only in before, new_col only in after -> should show as schema changes
    report = DataComparator().compare_dataframe(
        before, after, level="schema", columns=["old_col", "new_col"]
    )
    statuses = dict(zip(report["Metric"], report["Status"]))
    assert statuses["old_col"] == "Deleted Column"
    assert statuses["new_col"] == "New Column"


def test_compare_dataframe_columns_dataset_level_uses_full_frame(before_after):
    before, after = before_after
    report = DataComparator().compare_dataframe(
        before, after, level="dataset", columns=["age"]
    )
    cols_row = report[report["Metric"] == "Columns"].iloc[0]
    # Full frame has 3 columns on each side, not just the 1 selected
    assert cols_row["Before"] == 3
    assert cols_row["After"] == 3


def test_summary_columns_restricts_output(before_after):
    before, after = before_after
    summary = DataComparator().summary(before, after, columns=["age", "city"])
    assert set(summary["Column"]) == {"age", "city"}


def test_summary_columns_unknown_raises(before_after):
    before, after = before_after
    with pytest.raises(ValueError):
        DataComparator().summary(before, after, columns=["nonexistent"])
