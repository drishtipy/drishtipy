import numpy as np
import pandas as pd
import pytest

from drishtipy import DataProfiler, DataComparator


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 32, 47, np.nan, 29, 400],
            "salary": [50000, 60000, 75000, 52000, 61000, 59000],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Mumbai"],
        }
    )


# =====================================================
# DataProfiler.to_html
# =====================================================


def test_to_html_returns_string(sample_df):
    html = DataProfiler(sample_df).to_html()
    assert "<html" in html
    assert "Schema" in html
    assert "Statistics" in html


def test_to_html_single_section(sample_df):
    html = DataProfiler(sample_df).to_html(section="quality")
    assert "Quality" in html


def test_to_html_writes_file(sample_df, tmp_path):
    out = tmp_path / "report.html"
    html = DataProfiler(sample_df).to_html(path=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_to_html_custom_title(sample_df):
    html = DataProfiler(sample_df).to_html(title="My Custom Title")
    assert "My Custom Title" in html


# =====================================================
# DataProfiler.from_csv
# =====================================================


def test_from_csv_small_file_loads_fully(tmp_path):
    df = pd.DataFrame({"a": range(50), "b": [f"row{i}" for i in range(50)]})
    csv_path = tmp_path / "small.csv"
    df.to_csv(csv_path, index=False)

    profiler = DataProfiler.from_csv(str(csv_path), sample_size=1000, chunksize=10)

    assert profiler.is_sampled is False
    assert profiler.total_rows_in_source == 50
    assert len(profiler.df) == 50


def test_from_csv_large_file_samples(tmp_path):
    df = pd.DataFrame({"a": range(10_000), "b": ["x"] * 10_000})
    csv_path = tmp_path / "large.csv"
    df.to_csv(csv_path, index=False)

    profiler = DataProfiler.from_csv(
        str(csv_path), sample_size=500, chunksize=1000, random_state=42
    )

    assert profiler.is_sampled is True
    assert profiler.total_rows_in_source == 10_000
    # sampling is probabilistic, so allow some tolerance around 500
    assert 300 < len(profiler.df) < 900


def test_from_csv_sampled_profile_still_works(tmp_path):
    df = pd.DataFrame({"a": range(5000), "b": np.random.default_rng(0).random(5000)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    profiler = DataProfiler.from_csv(
        str(csv_path), sample_size=200, chunksize=500, random_state=1
    )
    report = profiler.info_dataframe(section="statistics")
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 2


def test_from_csv_html_mentions_sampling(tmp_path):
    df = pd.DataFrame({"a": range(5000)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    profiler = DataProfiler.from_csv(
        str(csv_path), sample_size=100, chunksize=500, random_state=1
    )
    html = profiler.to_html()
    assert "sample" in html.lower()
    assert "5,000" in html or "5000" in html


# =====================================================
# DataComparator.to_html
# =====================================================


def test_comparator_to_html_returns_string(sample_df):
    before = sample_df
    after = sample_df.fillna(30)

    html = DataComparator().to_html(before, after)
    assert "<html" in html
    assert "Summary" in html
    assert "Comparison" in html


def test_comparator_to_html_writes_file(sample_df, tmp_path):
    before = sample_df
    after = sample_df.fillna(30)
    out = tmp_path / "compare.html"

    html = DataComparator().to_html(before, after, path=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_comparator_to_html_respects_columns(sample_df):
    before = sample_df
    after = sample_df.fillna(30)

    html = DataComparator().to_html(before, after, columns=["age"])
    assert "age" in html
