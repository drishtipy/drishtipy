import numpy as np
import pandas as pd
import pytest

from drishtipy import DataProfiler


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 32, 47, np.nan, 29, 400],
            "salary": [50000, 60000, 75000, 52000, 61000, 59000],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", " Mumbai"],
            "id": [1, 2, 3, 4, 5, 6],
        }
    )


def test_requires_dataframe():
    with pytest.raises(TypeError):
        DataProfiler([1, 2, 3])


def test_invalid_section(sample_df):
    with pytest.raises(ValueError):
        DataProfiler(sample_df).info_dataframe(section="bogus")


def test_invalid_column_type(sample_df):
    with pytest.raises(ValueError):
        DataProfiler(sample_df).info_dataframe(column_type="bogus")


def test_all_sections_present(sample_df):
    report = DataProfiler(sample_df).info_dataframe()
    assert set(report.keys()) == {"Schema", "Statistics", "Quality", "ML", "ETL"}
    for section_df in report.values():
        assert isinstance(section_df, pd.DataFrame)
        assert len(section_df) == sample_df.shape[1]


# =====================================================
# PROFILE REPORT (info_dataframe(section="All") return type)
# =====================================================


def test_info_dataframe_all_returns_dict_subclass(sample_df):
    report = DataProfiler(sample_df).info_dataframe()
    assert isinstance(report, dict)


def test_profile_report_dict_access_backward_compatible(sample_df):
    report = DataProfiler(sample_df).info_dataframe()
    # every access pattern a plain dict supports must still work
    assert report["Schema"] is not None
    assert len(report) == 5
    assert list(report.keys()) == list(dict(report).keys())
    for k, v in report.items():
        assert isinstance(v, pd.DataFrame)
    assert dict(report).keys() == report.keys()


def test_profile_report_repr_is_dashboard_not_raw_dict(sample_df):
    report = DataProfiler(sample_df).info_dataframe()
    text = repr(report)
    assert "Quality Score" in text
    assert "Data Profile Report" in text
    assert "{'Schema':" not in text  # not falling back to raw dict repr


def test_profile_report_repr_html_renders_full_report(sample_df):
    report = DataProfiler(sample_df).info_dataframe()
    html = report._repr_html_()
    assert "<html" in html.lower()
    assert "Schema" in html
    assert "Quality" in html


def test_profile_report_repr_handles_single_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    report = DataProfiler(df).info_dataframe()
    text = repr(report)
    assert "1 column" in text
    assert "1 columns" not in text  # singular, not "1 column(s)" or plural


def test_profile_report_repr_no_alerts_message():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    report = DataProfiler(df).info_dataframe()
    assert "No alerts" in repr(report)


def test_profile_report_via_accessor(sample_df):
    report = sample_df.profile.info()
    assert isinstance(report, dict)
    assert "Quality Score" in repr(report)


def test_single_section_returns_dataframe(sample_df):
    schema = DataProfiler(sample_df).info_dataframe(section="Schema")
    assert isinstance(schema, pd.DataFrame)
    assert "Column" in schema.columns
    assert list(schema["Column"]) == list(sample_df.columns)


def test_case_insensitive_args(sample_df):
    a = DataProfiler(sample_df).info_dataframe(section="QUALITY")
    b = DataProfiler(sample_df).info_dataframe(section="quality")
    pd.testing.assert_frame_equal(a, b)


def test_column_type_numeric_filter(sample_df):
    ml = DataProfiler(sample_df).info_dataframe(section="ml", column_type="numeric")
    assert set(ml["Column"]) == {"age", "salary", "id"}


def test_column_type_categorical_filter(sample_df):
    ml = DataProfiler(sample_df).info_dataframe(
        section="ml", column_type="categorical"
    )
    assert set(ml["Column"]) == {"city"}


def test_column_type_filters_out_everything():
    df = pd.DataFrame({"only_text": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        DataProfiler(df).info_dataframe(column_type="numeric")


def test_schema_missing_counts(sample_df):
    schema = DataProfiler(sample_df).info_dataframe(section="schema")
    age_row = schema[schema["Column"] == "age"].iloc[0]
    assert age_row["Missing Count"] == 1
    assert age_row["Non-Null Count"] == 5


# =====================================================
# STATISTICS — extended dispersion / central tendency
# =====================================================


def test_statistics_rms_always_defined_for_numeric():
    df = pd.DataFrame({"x": [-5, -2, 0, 3, 7]})
    stats = DataProfiler(df).info_dataframe(section="statistics")
    row = stats[stats["Column"] == "x"].iloc[0]
    # RMS = sqrt(mean(x^2)), defined even with negative values
    assert row["RMS"] == round((sum(v ** 2 for v in [-5, -2, 0, 3, 7]) / 5) ** 0.5, 4)


def test_statistics_geometric_harmonic_mean_positive_only():
    df = pd.DataFrame({"positive": [1, 2, 4, 8], "mixed": [1, -2, 4, 8]})
    stats = DataProfiler(df).info_dataframe(section="statistics")

    pos_row = stats[stats["Column"] == "positive"].iloc[0]
    assert pos_row["Geometric Mean"] is not None
    assert pos_row["Harmonic Mean"] is not None

    mixed_row = stats[stats["Column"] == "mixed"].iloc[0]
    assert pd.isna(mixed_row["Geometric Mean"])
    assert pd.isna(mixed_row["Harmonic Mean"])


def test_statistics_cv_percent():
    df = pd.DataFrame({"x": [10, 10, 10, 10]})  # zero variance -> CV = 0
    stats = DataProfiler(df).info_dataframe(section="statistics")
    row = stats[stats["Column"] == "x"].iloc[0]
    assert row["CV %"] == 0.0


def test_statistics_cv_none_when_mean_zero():
    df = pd.DataFrame({"x": [-5, 5, -3, 3]})  # mean == 0
    stats = DataProfiler(df).info_dataframe(section="statistics")
    row = stats[stats["Column"] == "x"].iloc[0]
    assert pd.isna(row["CV %"])


def test_statistics_mad_matches_manual_calc():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    stats = DataProfiler(df).info_dataframe(section="statistics")
    row = stats[stats["Column"] == "x"].iloc[0]
    mean = sum([1, 2, 3, 4, 5]) / 5
    expected = sum(abs(v - mean) for v in [1, 2, 3, 4, 5]) / 5
    assert row["MAD"] == round(expected, 4)


def test_statistics_z_outlier_count_detects_extreme_value():
    df = pd.DataFrame({"x": [10] * 20 + [100]})  # one clear extreme value
    stats = DataProfiler(df).info_dataframe(section="statistics")
    row = stats[stats["Column"] == "x"].iloc[0]
    assert row["Z-Outlier Count"] >= 1


def test_statistics_non_numeric_extended_columns_are_none():
    df = pd.DataFrame({"city": ["Delhi", "Mumbai", "Pune"]})
    stats = DataProfiler(df).info_dataframe(section="statistics")
    row = stats[stats["Column"] == "city"].iloc[0]
    for col in ("Geometric Mean", "Harmonic Mean", "RMS", "CV %", "MAD"):
        assert pd.isna(row[col])


def test_quality_outlier_detection(sample_df):
    quality = DataProfiler(sample_df).info_dataframe(section="quality")
    age_row = quality[quality["Column"] == "age"].iloc[0]
    # 400 should register as an IQR outlier
    assert age_row["Outlier Count"] >= 1


def test_ml_possible_id(sample_df):
    ml = DataProfiler(sample_df).info_dataframe(section="ml")
    id_row = ml[ml["Column"] == "id"].iloc[0]
    assert id_row["Feature Status"] == "Possible ID"


def test_etl_flags_extra_spaces(sample_df):
    etl = DataProfiler(sample_df).info_dataframe(section="etl")
    city_row = etl[etl["Column"] == "city"].iloc[0]
    assert "Extra Spaces" in city_row["Issue"]
    assert city_row["ETL Status"] == "Needs Cleaning"


def test_etl_ready_status_for_clean_column():
    df = pd.DataFrame({"clean": [1, 2, 3, 4, 5]})
    etl = DataProfiler(df).info_dataframe(section="etl")
    row = etl.iloc[0]
    assert row["ETL Status"] == "Ready"
    assert row["Issue"] == "None"


def test_single_row_dataframe_no_zero_division():
    df = pd.DataFrame({"x": [1]})
    # len(s) == 1, guards against ZeroDivisionError should not trigger anyway
    report = DataProfiler(df).info_dataframe()
    assert report["Schema"]["Unique %"].iloc[0] == 100.0


# =====================================================
# QUALITY SCORE
# =====================================================


def test_quality_score_overall_shape(sample_df):
    result = DataProfiler(sample_df).quality_score()
    assert list(result["Metric"]) == [
        "Overall Quality Score",
        "Missing Values",
        "Duplicates",
        "Outliers",
        "Data Types",
        "Invalid Values",
    ]
    assert (result["Score"].between(0, 100)).all()


def test_quality_score_by_column_shape(sample_df):
    result = DataProfiler(sample_df).quality_score(by="column")
    assert list(result["Column"]) == list(sample_df.columns)
    for c in [
        "Missing Score",
        "Duplicate Score",
        "Outlier Score",
        "Data Type Score",
        "Invalid Score",
        "Quality Score",
    ]:
        assert (result[c].between(0, 100)).all()


def test_quality_score_perfect_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["p", "q", "r", "s", "t"]})
    result = DataProfiler(df).quality_score()
    overall = result.loc[result["Metric"] == "Overall Quality Score", "Score"].iloc[0]
    assert overall == 100.0


def test_quality_score_invalid_by(sample_df):
    with pytest.raises(ValueError):
        DataProfiler(sample_df).quality_score(by="bogus")


def test_quality_score_invalid_weights_key(sample_df):
    with pytest.raises(ValueError):
        DataProfiler(sample_df).quality_score(weights={"bogus": 10})


def test_quality_score_missing_values_penalized():
    clean = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    dirty = pd.DataFrame({"a": [1, 2, None, None, 5]})
    clean_score = DataProfiler(clean).quality_score()
    dirty_score = DataProfiler(dirty).quality_score()
    clean_overall = clean_score.loc[
        clean_score["Metric"] == "Overall Quality Score", "Score"
    ].iloc[0]
    dirty_overall = dirty_score.loc[
        dirty_score["Metric"] == "Overall Quality Score", "Score"
    ].iloc[0]
    assert dirty_overall < clean_overall


def test_quality_score_flags_numeric_stored_as_text():
    df = pd.DataFrame({"code": ["1", "2", "3", "4", "5"]})
    result = DataProfiler(df).quality_score(by="column")
    assert result.loc[0, "Data Type Score"] < 100


def test_quality_score_weights_shift_overall_score(sample_df):
    default = DataProfiler(sample_df).quality_score()
    reweighted = DataProfiler(sample_df).quality_score(
        weights={"missing": 100, "duplicates": 0, "outliers": 0, "dtypes": 0, "invalid": 0}
    )
    default_overall = default.loc[
        default["Metric"] == "Overall Quality Score", "Score"
    ].iloc[0]
    reweighted_overall = reweighted.loc[
        reweighted["Metric"] == "Overall Quality Score", "Score"
    ].iloc[0]
    missing_only = reweighted.loc[reweighted["Metric"] == "Missing Values", "Score"].iloc[0]
    assert reweighted_overall == missing_only
    assert reweighted_overall != default_overall


# =====================================================
# PII DETECTION
# =====================================================


@pytest.fixture
def pii_df():
    return pd.DataFrame(
        {
            "email": [
                "john.doe@gmail.com",
                "jane_smith@yahoo.com",
                "bob.k@company.co.in",
                "x@y.com",
                "not_an_email",
            ],
            "mobile": [
                "9876543210",
                "+91 9123456780",
                "9012345678",
                "98765 43210",
                "12345",
            ],
            "pan_no": [
                "ABCDE1234F",
                "PQRSX5678Y",
                "LMNOP9999Z",
                "AAAAA0000A",
                "notapan",
            ],
            "aadhaar": [
                "1234 5678 9012",
                "9876 5432 1098",
                "1111 2222 3333",
                "0000 1111 2222",
                "abc",
            ],
            "ip_col": [
                "192.168.1.1",
                "10.0.0.254",
                "8.8.8.8",
                "256.1.1.1",
                "not.an.ip",
            ],
            "full_name": [
                "Rahul Sharma",
                "Priya Verma",
                "Amit Kumar",
                "Sunita Rao",
                "John",
            ],
            "address": [
                "221B Baker Street, London",
                "12 MG Road, Bangalore",
                "45 Nagar Colony",
                "Flat 9, Sector 21",
                "No address",
            ],
            "salary": [50000, 60000, 75000, 52000, 61000],
            "city": ["Delhi", "Mumbai", "Pune", "Delhi", "Chennai"],
        }
    )


def test_pii_detects_expected_types(pii_df):
    report = DataProfiler(pii_df).pii()
    detected = dict(zip(report["Column"], report["PII Type"]))
    assert detected["email"] == "EMAIL"
    assert detected["mobile"] == "PHONE"
    assert detected["pan_no"] == "POSSIBLE_PAN"
    assert detected["aadhaar"] == "POSSIBLE_ID"
    assert detected["ip_col"] == "IP"
    assert detected["full_name"] == "POSSIBLE_NAME"
    assert detected["address"] == "POSSIBLE_ADDRESS"


def test_pii_does_not_flag_clean_columns(pii_df):
    report = DataProfiler(pii_df).pii()
    assert "salary" not in set(report["Column"])
    assert "city" not in set(report["Column"])


def test_pii_report_sorted_by_confidence_desc(pii_df):
    report = DataProfiler(pii_df).pii()
    scores = list(report["Confidence %"])
    assert scores == sorted(scores, reverse=True)


def test_pii_min_confidence_filters_out_weak_matches(pii_df):
    strict = DataProfiler(pii_df).pii(min_confidence=99.9)
    assert len(strict) <= len(DataProfiler(pii_df).pii())


def test_pii_invalid_min_confidence(pii_df):
    with pytest.raises(ValueError):
        DataProfiler(pii_df).pii(min_confidence=150)


def test_pii_no_matches_returns_empty_dataframe():
    df = pd.DataFrame({"age": [1, 2, 3], "category": ["a", "b", "c"]})
    report = DataProfiler(df).pii()
    assert isinstance(report, pd.DataFrame)
    assert report.empty
    assert list(report.columns) == ["Column", "PII Type", "Confidence %"]


def test_pii_mask_returns_dataframe_same_shape(pii_df):
    masked = DataProfiler(pii_df).pii(mask=True)
    assert isinstance(masked, pd.DataFrame)
    assert masked.shape == pii_df.shape


def test_pii_mask_email_preserves_domain(pii_df):
    masked = DataProfiler(pii_df).pii(mask=True)
    assert masked["email"].iloc[0].endswith("@gmail.com")
    assert masked["email"].iloc[0] != pii_df["email"].iloc[0]


def test_pii_mask_leaves_non_pii_columns_untouched(pii_df):
    masked = DataProfiler(pii_df).pii(mask=True)
    pd.testing.assert_series_equal(masked["salary"], pii_df["salary"])
    pd.testing.assert_series_equal(masked["city"], pii_df["city"])


def test_pii_mask_address_is_fully_redacted(pii_df):
    masked = DataProfiler(pii_df).pii(mask=True)
    assert (masked["address"] == "[REDACTED ADDRESS]").all()


def test_pii_numeric_phone_column_detected():
    df = pd.DataFrame({"mobile": [9876543210, 9123456780, 9012345678]})
    report = DataProfiler(df).pii()
    assert report.loc[0, "PII Type"] == "PHONE"


def test_pii_handles_all_null_column():
    df = pd.DataFrame({"a": [None, None, None]})
    report = DataProfiler(df).pii()
    assert report.empty


# =====================================================
# CORRELATIONS
# =====================================================


@pytest.fixture
def corr_df():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(0, 1, n)
    b = a * 2 + rng.normal(0, 0.05, n)  # strongly correlated with a
    c = rng.normal(0, 1, n)  # independent
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_correlations_matrix_shape(corr_df):
    matrix = DataProfiler(corr_df).correlations()
    assert list(matrix.columns) == ["a", "b", "c"]
    assert list(matrix.index) == ["a", "b", "c"]
    assert matrix.loc["a", "a"] == 1.0


def test_correlations_pairs_shape(corr_df):
    pairs = DataProfiler(corr_df).correlations(by="pairs")
    assert set(pairs.columns) == {"Column A", "Column B", "Correlation", "R²"}
    assert len(pairs) == 3  # 3 choose 2


def test_correlations_pairs_sorted_by_strength(corr_df):
    pairs = DataProfiler(corr_df).correlations(by="pairs")
    abs_vals = pairs["Correlation"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_correlations_detects_strong_pair(corr_df):
    pairs = DataProfiler(corr_df).correlations(by="pairs")
    top = pairs.iloc[0]
    assert {top["Column A"], top["Column B"]} == {"a", "b"}
    assert top["Correlation"] > 0.9


def test_correlations_r_squared_matches_correlation_squared(corr_df):
    pairs = DataProfiler(corr_df).correlations(by="pairs")
    for _, row in pairs.iterrows():
        assert row["R²"] == round(row["Correlation"] ** 2, 4)


def test_correlations_threshold_filters_weak_pairs(corr_df):
    pairs = DataProfiler(corr_df).correlations(by="pairs", threshold=0.9)
    assert len(pairs) == 1
    assert pairs.iloc[0]["Correlation"] > 0.9


def test_correlations_requires_two_numeric_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        DataProfiler(df).correlations()


def test_correlations_invalid_method(corr_df):
    with pytest.raises(ValueError):
        DataProfiler(corr_df).correlations(method="bogus")


def test_correlations_invalid_by(corr_df):
    with pytest.raises(ValueError):
        DataProfiler(corr_df).correlations(by="bogus")


def test_correlations_spearman_method_runs(corr_df):
    matrix = DataProfiler(corr_df).correlations(method="SPEARMAN")
    assert matrix.loc["a", "a"] == 1.0


# =====================================================
# ALERTS
# =====================================================


@pytest.fixture
def alerts_df():
    return pd.DataFrame(
        {
            "age": [25, 32, 47, np.nan, 29, 400],
            "constant_col": [5, 5, 5, 5, 5, 5],
            "email": [
                "a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "f@x.com"
            ],
            "code_as_text": ["1", "2", "3", "4", "5", "6"],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Mumbai"],
        }
    )


def test_alerts_returns_dataframe_with_expected_columns(alerts_df):
    result = DataProfiler(alerts_df).alerts()
    assert list(result.columns) == ["Severity", "Column", "Alert", "Details"]


def test_alerts_sorted_by_severity(alerts_df):
    result = DataProfiler(alerts_df).alerts()
    order = {"High": 0, "Medium": 1, "Low": 2}
    ranks = [order[s] for s in result["Severity"]]
    assert ranks == sorted(ranks)


def test_alerts_flags_constant_column(alerts_df):
    result = DataProfiler(alerts_df).alerts()
    hit = result[(result["Column"] == "constant_col") & (result["Alert"] == "Constant Column")]
    assert len(hit) == 1
    assert hit.iloc[0]["Severity"] == "High"


def test_alerts_flags_missing_values(alerts_df):
    result = DataProfiler(alerts_df).alerts()
    hit = result[(result["Column"] == "age") & (result["Alert"] == "Missing Values")]
    assert len(hit) == 1


def test_alerts_flags_pii(alerts_df):
    result = DataProfiler(alerts_df).alerts()
    hit = result[(result["Column"] == "email") & (result["Alert"] == "Potential PII")]
    assert len(hit) == 1
    assert hit.iloc[0]["Severity"] == "High"


def test_alerts_flags_wrong_data_type(alerts_df):
    result = DataProfiler(alerts_df).alerts()
    hit = result[
        (result["Column"] == "code_as_text") & (result["Alert"] == "Wrong Data Type")
    ]
    assert len(hit) == 1


def test_alerts_min_severity_filters(alerts_df):
    all_alerts = DataProfiler(alerts_df).alerts()
    high_only = DataProfiler(alerts_df).alerts(min_severity="high")
    assert len(high_only) <= len(all_alerts)
    assert set(high_only["Severity"]) <= {"High"}


def test_alerts_invalid_min_severity(alerts_df):
    with pytest.raises(ValueError):
        DataProfiler(alerts_df).alerts(min_severity="bogus")


def test_alerts_no_findings_returns_empty_dataframe():
    df = pd.DataFrame({"clean": [1, 2, 3, 4, 5]})
    result = DataProfiler(df).alerts()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Severity", "Column", "Alert", "Details"]


def test_alerts_flags_duplicate_rows():
    df = pd.DataFrame({"x": [1, 2, 1, 2, 1, 2] * 5})
    result = DataProfiler(df).alerts()
    hit = result[(result["Column"] == "(table)") & (result["Alert"] == "Duplicate Rows")]
    assert len(hit) == 1


def test_alerts_flags_highly_correlated_pair():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(0, 1, n)
    b = a * 2 + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"a": a, "b": b})
    result = DataProfiler(df).alerts()
    hit = result[result["Alert"] == "Highly Correlated Pair"]
    assert len(hit) == 1
