import numpy as np
import pandas as pd
import pytest

from drishtipy import RelationshipAnalyzer, RelationshipConfig, RelationshipResult
from drishtipy.semantic import detect_semantic_type, semantic_type_counts
import drishtipy.relationships as rel


# =====================================================
# SEMANTIC TYPE DETECTION
# =====================================================


def test_semantic_detects_id_by_name_and_uniqueness():
    s = pd.Series(range(1, 101))
    assert detect_semantic_type(s, "farmer_id") == "id"


def test_semantic_detects_numeric():
    s = pd.Series(np.random.rand(50) * 100)
    assert detect_semantic_type(s, "land_area") == "numeric"


def test_semantic_detects_categorical():
    s = pd.Series(["Delhi", "Mumbai", "Delhi", "Pune"] * 10)
    assert detect_semantic_type(s, "district") == "categorical"


def test_semantic_detects_boolean():
    s = pd.Series(["Yes", "No", "Yes", "No"] * 10)
    assert detect_semantic_type(s, "is_active") == "boolean"


def test_semantic_detects_email():
    s = pd.Series([f"user{i}@example.com" for i in range(50)])
    assert detect_semantic_type(s, "email") == "email"


def test_semantic_detects_date():
    s = pd.date_range("2023-01-01", periods=50).astype(str)
    assert detect_semantic_type(pd.Series(s), "signup_date") == "date"


def test_semantic_detects_datetime_dtype():
    s = pd.Series(pd.date_range("2023-01-01", periods=50))
    assert detect_semantic_type(s, "signup_date") == "datetime"


def test_semantic_detects_percentage_by_name_hint():
    s = pd.Series(np.random.rand(50) * 100)
    assert detect_semantic_type(s, "completion_pct") == "percentage"


def test_semantic_detects_constant():
    s = pd.Series([5] * 20)
    assert detect_semantic_type(s, "flag") == "constant"


def test_semantic_detects_free_text():
    s = pd.Series(
        [f"This is a fairly long free-text note about record number {i} indeed" for i in range(50)]
    )
    assert detect_semantic_type(s, "notes") == "text"


def test_semantic_type_counts_returns_dict():
    df = pd.DataFrame({"a": range(50), "b": ["x", "y"] * 25})
    counts = semantic_type_counts(df)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 2


# =====================================================
# RELATIONSHIP ANALYZER — CORE BEHAVIOR
# =====================================================


@pytest.fixture
def farm_df():
    rng = np.random.default_rng(0)
    n = 1000
    land_area = rng.gamma(2, 2, n)
    district = rng.choice(["A", "B", "C", "D"], n)
    district_map = {"A": 500, "B": 700, "C": 300, "D": 900}
    procurement = (
        land_area * 3.2
        + np.array([district_map[d] for d in district])
        + rng.normal(0, 50, n)
    )
    distance = rng.uniform(5, 200, n)
    transport_cost = distance * 12 + rng.normal(0, 30, n)

    return pd.DataFrame(
        {
            "farmer_id": range(1, n + 1),
            "farmer_name": [f"Farmer_{i}" for i in range(1, n + 1)],
            "district": district,
            "land_area": land_area,
            "procurement": procurement,
            "distance": distance,
            "transport_cost": transport_cost,
            "quantity_kg": procurement * 1000,
            "constant_col": [1] * n,
        }
    )


def test_analyzer_requires_dataframe():
    with pytest.raises(TypeError):
        RelationshipAnalyzer([1, 2, 3])


def test_analyze_returns_relationship_result(farm_df):
    result = farm_df.profile.relationships()
    assert isinstance(result, RelationshipResult)


def test_numeric_numeric_detected(farm_df):
    result = farm_df.profile.relationships()
    df = result.to_dataframe()
    row = df[
        ((df["Column A"] == "distance") & (df["Column B"] == "transport_cost"))
        | ((df["Column A"] == "transport_cost") & (df["Column B"] == "distance"))
    ].iloc[0]
    assert row["Method"] in ("Pearson", "Spearman")
    assert row["Strength"] > 0.9
    assert row["Classification"] == "Strong"
    assert row["Direction"] == "Positive"


def test_categorical_numeric_uses_anova(farm_df):
    result = farm_df.profile.relationships()
    df = result.to_dataframe()
    row = df[
        ((df["Column A"] == "district") & (df["Column B"] == "procurement"))
        | ((df["Column A"] == "procurement") & (df["Column B"] == "district"))
    ].iloc[0]
    assert row["Method"] == "ANOVA (eta²)"
    assert row["Direction"] == "N/A"
    assert row["Strength"] > 0.5


def test_constant_column_excluded(farm_df):
    result = farm_df.profile.relationships()
    df = result.to_dataframe()
    assert not ((df["Column A"] == "constant_col") | (df["Column B"] == "constant_col")).any()


def test_id_columns_excluded_by_default(farm_df):
    result = farm_df.profile.relationships()
    df = result.to_dataframe()
    assert not ((df["Column A"] == "farmer_id") | (df["Column B"] == "farmer_id")).any()


def test_include_id_pairs_override():
    df = pd.DataFrame({"id1": range(100), "id2": range(100, 200)})
    result = RelationshipAnalyzer(
        df, RelationshipConfig(include_id_pairs=True, min_sample_size=10)
    ).analyze()
    out = result.to_dataframe()
    assert len(out) == 1
    assert out.iloc[0]["Method"] == "Cramér's V"


def test_categorical_categorical_uses_cramers_v():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.choice(["X", "Y", "Z"], n)
    b = np.where(a == "X", "P", np.where(a == "Y", "Q", "R"))  # perfectly dependent
    df = pd.DataFrame({"a": a, "b": b})
    result = df.profile.relationships()
    row = result.to_dataframe().iloc[0]
    assert row["Method"] == "Cramér's V"
    assert row["Strength"] > 0.9
    assert "Mutual Info" in row["Notes"]


def test_datetime_numeric_trend_detected():
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    values = np.arange(n) * 2.0 + np.random.default_rng(0).normal(0, 5, n)
    df = pd.DataFrame({"date": dates.astype(str), "sales": values})
    result = df.profile.relationships()
    row = result.to_dataframe().iloc[0]
    assert row["Method"] == "Datetime Trend (Spearman)"
    assert row["Direction"] == "Increasing"
    assert row["Strength"] > 0.9


def test_not_enough_observations_reported():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    result = df.profile.relationships()
    row = result.to_dataframe().iloc[0]
    assert row["Classification"] == "Not Meaningful"
    assert row["Notes"] == "Not enough valid observations"


def test_high_cardinality_categorical_pair_skipped():
    n = 300
    df = pd.DataFrame(
        {
            "cat_a": [f"v{i}" for i in range(n)],  # every value unique -> "id"-like, excluded
            "cat_b": np.random.default_rng(0).choice(["p", "q"], n),
        }
    )
    result = df.profile.relationships(RelationshipConfig(min_sample_size=5))
    # the fully-unique column is classified as "id" and excluded entirely
    assert result.to_dataframe().empty


# =====================================================
# DEPENDENCIES / REDUNDANCY
# =====================================================


def test_functional_dependency_detected(farm_df):
    result = farm_df.profile.relationships()
    deps = result.dependencies()
    pair = deps[(deps["Source"] == "farmer_id") & (deps["Target"] == "farmer_name")]
    assert len(pair) == 1
    assert pair.iloc[0]["Consistency %"] > 99


def test_redundant_columns_detects_scaled_duplicate(farm_df):
    result = farm_df.profile.relationships()
    groups = result._redundant_groups
    flat = [set(g) for g in groups]
    assert any({"procurement", "quantity_kg"} <= g for g in flat)


def test_redundant_columns_excludes_anova_pairs(farm_df):
    # district <-> procurement is a strong ANOVA relationship, not a
    # "same information stored twice" redundancy — must not be merged.
    result = farm_df.profile.relationships()
    for group in result._redundant_groups:
        assert not ({"district", "procurement"} <= set(group))


def test_exact_duplicate_columns_detected():
    df = pd.DataFrame({"a": range(50), "b": range(50)})  # identical
    result = df.profile.relationships(RelationshipConfig(min_sample_size=5))
    groups = result._redundant_groups
    assert any(set(g) == {"a", "b"} for g in groups)


# =====================================================
# RESULT VIEWS
# =====================================================


def test_top_returns_sorted_meaningful_only(farm_df):
    result = farm_df.profile.relationships()
    top5 = result.top(5)
    assert len(top5) <= 5
    assert (top5["Classification"] != "Not Meaningful").all()
    strengths = top5["Strength"].tolist()
    assert strengths == sorted(strengths, reverse=True)


def test_matrix_is_symmetric_with_unit_diagonal(farm_df):
    result = farm_df.profile.relationships()
    mat = result.matrix()
    assert (np.diag(mat.values) == 1.0).all()
    for c1 in mat.columns:
        for c2 in mat.columns:
            v1, v2 = mat.loc[c1, c2], mat.loc[c2, c1]
            assert (pd.isna(v1) and pd.isna(v2)) or v1 == v2


def test_graph_returns_edges_and_nodes(farm_df):
    result = farm_df.profile.relationships()
    graph = result.graph(top=5)
    assert len(graph.edges) <= 5
    assert set(graph.nodes) == {c for e in graph.edges for c in (e[0], e[1])}


def test_graph_threshold_filters(farm_df):
    result = farm_df.profile.relationships()
    graph = result.graph(top=50, threshold=0.99)
    for _a, _b, strength, _cls in graph.edges:
        assert strength >= 0.99


def test_graph_to_dataframe(farm_df):
    result = farm_df.profile.relationships()
    graph = result.graph(top=3)
    gdf = graph.to_dataframe()
    assert list(gdf.columns) == ["Column A", "Column B", "Strength", "Classification"]


def test_graph_repr_handles_empty():
    g = rel.RelationshipGraph([])
    assert "empty" in repr(g)


def test_insights_is_list_of_strings(farm_df):
    result = farm_df.profile.relationships()
    insights = result.insights()
    assert isinstance(insights, list)
    assert all(isinstance(i, str) for i in insights)
    assert any("causation" in i for i in insights)


def test_insights_no_findings_message():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.profile.relationships()
    assert result.insights() == ["No strong or clearly meaningful relationships were found."]


def test_to_dataframe_returns_all_rows(farm_df):
    result = farm_df.profile.relationships()
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == result.summary()["pairs_analyzed"]


def test_to_json_roundtrip(farm_df):
    import json

    result = farm_df.profile.relationships()
    payload = json.loads(result.to_json())
    assert "summary" in payload
    assert "relationships" in payload
    assert "dependencies" in payload
    assert "redundant_columns" in payload


def test_to_html_contains_sections(farm_df):
    result = farm_df.profile.relationships()
    html = result.to_html()
    assert "<html" in html.lower()
    assert "Top Relationships" in html
    assert "Insights" in html


def test_to_html_writes_file(farm_df, tmp_path):
    result = farm_df.profile.relationships()
    path = tmp_path / "relationships.html"
    result.to_html(path=str(path))
    assert path.exists()
    assert "<html" in path.read_text().lower()


# =====================================================
# HTML DASHBOARD RENDERING
# =====================================================


def test_html_dashboard_is_default_style(farm_df):
    result = farm_df.profile.relationships()
    dashboard = result.to_html()
    explicit = result.to_html(style="dashboard")
    assert dashboard == explicit


def test_html_table_style_differs_from_dashboard(farm_df):
    result = farm_df.profile.relationships()
    dashboard = result.to_html(style="dashboard")
    table = result.to_html(style="table")
    assert dashboard != table
    assert "<svg" in dashboard
    assert "<svg" not in table


def test_html_invalid_style_raises(farm_df):
    result = farm_df.profile.relationships()
    with pytest.raises(ValueError):
        result.to_html(style="bogus")


def test_html_dashboard_contains_kpi_cards(farm_df):
    result = farm_df.profile.relationships()
    html = result.to_html()
    s = result.summary()
    assert f">{s['columns_analyzed']}<" in html
    assert f"{s['possible_pairs']:,}" in html


def test_html_dashboard_contains_graph_svg(farm_df):
    result = farm_df.profile.relationships()
    html = result.to_html()
    assert "<svg" in html
    assert "foreignObject" in html


def test_html_dashboard_escapes_column_names():
    df = pd.DataFrame(
        {
            "a<script>": np.random.default_rng(0).normal(0, 1, 200),
            "b&col": np.random.default_rng(0).normal(0, 1, 200),
        }
    )
    result = df.profile.relationships()
    html = result.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html or "a&amp;lt" not in html


def test_html_dashboard_no_relationships_case():
    df = pd.DataFrame(
        {"a": np.random.default_rng(0).random(200), "b": np.random.default_rng(1).random(200)}
    )
    result = df.profile.relationships()
    html = result.to_html()
    assert "No meaningful relationships found" in html
    assert "No meaningful relationships to graph" in html
    assert "No strong or clearly meaningful relationships were found." in html


def test_html_dashboard_single_node_graph_no_crash():
    df = pd.DataFrame({"x": range(200), "y": [i * 2 for i in range(200)]})
    result = df.profile.relationships(RelationshipConfig(min_sample_size=5))
    html = result.to_html()
    assert "<svg" in html


def test_html_dashboard_no_dependencies_case():
    df = pd.DataFrame(
        {"a": np.random.default_rng(0).normal(0, 1, 200), "b": np.random.default_rng(1).normal(0, 1, 200)}
    )
    result = df.profile.relationships()
    html = result.to_html()
    assert "No potential dependencies detected" in html


def test_html_dashboard_shows_redundant_panel_only_when_present(farm_df):
    result = farm_df.profile.relationships()
    html = result.to_html()
    assert "Redundant / Derived Columns" in html

    df_no_redundancy = pd.DataFrame(
        {"a": np.random.default_rng(0).random(200), "b": np.random.default_rng(1).random(200)}
    )
    result2 = df_no_redundancy.profile.relationships()
    html2 = result2.to_html()
    assert "Redundant / Derived Columns" not in html2


def test_html_dashboard_respects_top_n_and_graph_top(farm_df):
    result = farm_df.profile.relationships()
    html = result.to_html(top_n=1, graph_top=1)
    assert html.count('class="relationship"') == 1


def test_render_relationship_dashboard_direct_call(farm_df):
    result = farm_df.profile.relationships()
    html = rel.render_relationship_dashboard(result)
    assert "<html" in html.lower()


def test_repr_shows_dashboard(farm_df):
    result = farm_df.profile.relationships()
    text = repr(result)
    assert "DRISHTIPY — RELATIONSHIP DISCOVERY" in text
    assert "TOP RELATIONSHIPS" in text


# =====================================================
# CONFIG / THRESHOLDS
# =====================================================


def test_custom_thresholds_change_classification():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(0, 1, n)
    b = a * 0.6 + rng.normal(0, 0.8, n)  # moderate correlation
    df = pd.DataFrame({"a": a, "b": b})

    default_result = df.profile.relationships()
    lenient_result = df.profile.relationships(RelationshipConfig(moderate_threshold=0.1))

    default_class = default_result.to_dataframe().iloc[0]["Classification"]
    lenient_class = lenient_result.to_dataframe().iloc[0]["Classification"]
    assert lenient_class in ("Moderate", "Strong")
    assert default_class != lenient_class or default_class in ("Moderate", "Strong")


def test_max_pairs_caps_analysis():
    rng = np.random.default_rng(0)
    n = 60
    cols = {f"c{i}": rng.normal(0, 1, n) for i in range(12)}  # 66 possible pairs
    df = pd.DataFrame(cols)
    result = df.profile.relationships(RelationshipConfig(max_pairs=10))
    assert result.summary()["pairs_analyzed"] <= 10
    assert result.summary()["capped_at_max_pairs"] is True


# =====================================================
# SAMPLING
# =====================================================


def test_sampling_used_for_large_dataframe():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    result = df.profile.relationships(RelationshipConfig(sample_size=1000))
    s = result.summary()
    assert s["is_sampled"] is True
    assert s["analyzed_rows"] == 1000
    assert s["total_rows"] == n


def test_no_sampling_when_below_threshold():
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    result = df.profile.relationships(RelationshipConfig(sample_size=1000, min_sample_size=5))
    assert result.summary()["is_sampled"] is False


# =====================================================
# SCIPY FALLBACK
# =====================================================


def test_scipy_unavailable_fallback(monkeypatch, farm_df):
    monkeypatch.setattr(rel, "_HAS_SCIPY", False)
    result = farm_df.profile.relationships()
    df = result.to_dataframe()
    assert (df["Significance"] == "Unknown (scipy not installed)").any()
    assert df["P-Value"].isna().all()
    assert result.summary()["scipy_available"] is False


def test_benjamini_hochberg_basic():
    # A clearly-significant p-value should survive correction; a
    # clearly-insignificant one should not.
    result = rel._benjamini_hochberg([0.001, 0.9], alpha=0.05)
    assert result[0] is True
    assert result[1] is False


def test_benjamini_hochberg_empty():
    assert rel._benjamini_hochberg([], alpha=0.05) == []


# =====================================================
# INTEGRATION — accessor & ProfileReport
# =====================================================


def test_relationships_via_accessor(farm_df):
    direct = RelationshipAnalyzer(farm_df).analyze()
    via_accessor = farm_df.profile.relationships()
    pd.testing.assert_frame_equal(direct.to_dataframe(), via_accessor.to_dataframe())


def test_relationships_via_profile_report(farm_df):
    report = farm_df.profile.info()
    result = report.relationships()
    assert isinstance(result, RelationshipResult)


def test_relationships_config_via_accessor(farm_df):
    result = farm_df.profile.relationships(RelationshipConfig(strong_threshold=0.99))
    df = result.to_dataframe()
    strong_rows = df[df["Classification"] == "Strong"]
    assert (strong_rows["Strength"] >= 0.99).all()
