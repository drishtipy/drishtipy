import numpy as np
import pandas as pd
import pytest

import drishtipy  # noqa: F401  (registers df.profile)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "age": [25, 32, 47, np.nan, 29, 400],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Mumbai"],
        }
    )


def test_accessor_is_registered(df):
    assert hasattr(df, "profile")


def test_info_matches_drishtipy(df):
    from drishtipy import DataProfiler

    direct = DataProfiler(df).info_dataframe(section="schema")
    via_accessor = df.profile.schema()
    pd.testing.assert_frame_equal(direct, via_accessor)


def test_info_returns_full_dict(df):
    report = df.profile.info()
    assert set(report.keys()) == {"Schema", "Statistics", "Quality", "ML", "ETL"}


def test_each_section_method(df):
    assert isinstance(df.profile.schema(), pd.DataFrame)
    assert isinstance(df.profile.statistics(), pd.DataFrame)
    assert isinstance(df.profile.quality(), pd.DataFrame)
    assert isinstance(df.profile.ml(), pd.DataFrame)
    assert isinstance(df.profile.etl(), pd.DataFrame)


def test_html_via_accessor(df, tmp_path):
    out = tmp_path / "r.html"
    html = df.profile.html(path=str(out))
    assert "<html" in html
    assert out.exists()


def test_compare_via_accessor(df):
    after = df.fillna(30)
    report = df.profile.compare(after, level="quality")
    assert isinstance(report, pd.DataFrame)
    age_missing = report[
        (report["Section"] == "Missing") & (report["Metric"] == "age")
    ].iloc[0]
    assert age_missing["Before"] == 1
    assert age_missing["After"] == 0


def test_compare_summary_via_accessor(df):
    after = df.fillna(30)
    summary = df.profile.compare_summary(after)
    assert set(summary["Column"]) == {"age", "city"}


def test_compare_html_via_accessor(df, tmp_path):
    after = df.fillna(30)
    out = tmp_path / "cmp.html"
    html = df.profile.compare_html(after, path=str(out))
    assert "<html" in html
    assert out.exists()


def test_quality_score_via_accessor(df):
    from drishtipy import DataProfiler

    direct = DataProfiler(df).quality_score()
    via_accessor = df.profile.quality_score()
    pd.testing.assert_frame_equal(direct, via_accessor)


def test_quality_score_by_column_via_accessor(df):
    result = df.profile.quality_score(by="column")
    assert isinstance(result, pd.DataFrame)
    assert list(result["Column"]) == list(df.columns)


def test_pii_via_accessor():
    from drishtipy import DataProfiler

    pii_df = pd.DataFrame({"email": ["a@b.com", "c@d.com", "e@f.com"]})
    direct = DataProfiler(pii_df).pii()
    via_accessor = pii_df.profile.pii()
    pd.testing.assert_frame_equal(direct, via_accessor)


def test_pii_mask_via_accessor():
    pii_df = pd.DataFrame({"email": ["a@b.com", "c@d.com", "e@f.com"]})
    masked = pii_df.profile.pii(mask=True)
    assert isinstance(masked, pd.DataFrame)
    assert masked["email"].iloc[0] != pii_df["email"].iloc[0]


def test_correlations_via_accessor():
    from drishtipy import DataProfiler

    rng = np.random.default_rng(0)
    n = 50
    a = rng.normal(0, 1, n)
    b = a * 2 + rng.normal(0, 0.05, n)
    corr_df = pd.DataFrame({"a": a, "b": b})

    direct = DataProfiler(corr_df).correlations()
    via_accessor = corr_df.profile.correlations()
    pd.testing.assert_frame_equal(direct, via_accessor)


def test_correlations_pairs_via_accessor():
    rng = np.random.default_rng(0)
    n = 50
    a = rng.normal(0, 1, n)
    b = a * 2 + rng.normal(0, 0.05, n)
    corr_df = pd.DataFrame({"a": a, "b": b})
    result = corr_df.profile.correlations(by="pairs", threshold=0.5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_alerts_via_accessor():
    from drishtipy import DataProfiler

    alerts_df = pd.DataFrame({"constant_col": [5, 5, 5, 5, 5]})
    direct = DataProfiler(alerts_df).alerts()
    via_accessor = alerts_df.profile.alerts()
    pd.testing.assert_frame_equal(direct, via_accessor)


def test_alerts_min_severity_via_accessor():
    alerts_df = pd.DataFrame({"constant_col": [5, 5, 5, 5, 5]})
    result = alerts_df.profile.alerts(min_severity="high")
    assert isinstance(result, pd.DataFrame)
    assert set(result["Severity"]) <= {"High"}


def test_accessor_reflects_mutation(df):
    # Accessor should always reflect the current state of the DataFrame
    schema_before = df.profile.schema()
    df["new_col"] = 1
    schema_after = df.profile.schema()
    assert len(schema_after) == len(schema_before) + 1
