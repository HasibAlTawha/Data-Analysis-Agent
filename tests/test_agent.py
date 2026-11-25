"""
Tests for Automated Data Analysis Agent
Run with: pytest tests/ -v
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.data_loader import load_csv, get_dataframe_summary, _clean_dataframe
from src.agent import set_dataframe, get_basic_stats, get_column_info, detect_outliers, get_value_counts


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """A small mixed-type dataframe for testing."""
    return pd.DataFrame({
        "id":       range(1, 101),
        "sales":    np.random.normal(500, 100, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date":     pd.date_range("2024-01-01", periods=100, freq="D"),
        "notes":    [f"note {i}" for i in range(100)],
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": ["x", None, "z", None, "w"],
        "c": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def df_with_outliers():
    data = list(range(1, 50)) + [1000, -500]   # obvious outliers
    return pd.DataFrame({"values": data, "group": ["A"] * 51})


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_clean_column_names(self):
        df = pd.DataFrame({"  col A  ": [1, 2], " col B": [3, 4]})
        cleaned = _clean_dataframe(df)
        assert "col A" in cleaned.columns
        assert "col B" in cleaned.columns

    def test_clean_string_values(self):
        df = pd.DataFrame({"name": ["  alice  ", " bob "]})
        cleaned = _clean_dataframe(df)
        assert cleaned["name"].tolist() == ["alice", "bob"]

    def test_summary_shape(self, sample_df):
        summary = get_dataframe_summary(sample_df)
        assert "100" in summary       # row count
        assert "rows" in summary.lower()

    def test_summary_contains_columns(self, sample_df):
        summary = get_dataframe_summary(sample_df)
        for col in sample_df.columns:
            assert col in summary

    def test_summary_null_info(self, df_with_nulls):
        summary = get_dataframe_summary(df_with_nulls)
        assert "null" in summary.lower() or "%" in summary

    def test_load_csv_file_not_found(self):
        with pytest.raises(Exception):
            load_csv("/nonexistent/path/file.csv")


# ── Agent tool tests ──────────────────────────────────────────────────────────

class TestAgentTools:
    def test_basic_stats_returns_string(self, sample_df):
        set_dataframe(sample_df)
        result = get_basic_stats.func("")
        assert isinstance(result, str)
        assert "sales" in result        # numeric column should appear

    def test_basic_stats_no_df(self):
        set_dataframe(None)
        result = get_basic_stats.func("")
        assert "No dataframe" in result

    def test_column_info_returns_string(self, sample_df):
        set_dataframe(sample_df)
        result = get_column_info.func("")
        assert isinstance(result, str)
        assert "Null" in result or "null" in result

    def test_detect_outliers_finds_outliers(self, df_with_outliers):
        set_dataframe(df_with_outliers)
        result = detect_outliers.func("")
        assert "outlier" in result.lower()
        assert "values" in result       # column name

    def test_detect_outliers_no_outliers(self):
        df = pd.DataFrame({"x": list(range(1, 51))})  # clean data
        set_dataframe(df)
        result = detect_outliers.func("")
        # Should either find outliers or state none
        assert isinstance(result, str)

    def test_value_counts_categorical(self, sample_df):
        set_dataframe(sample_df)
        result = get_value_counts.func("")
        assert isinstance(result, str)
        # Category column has A/B/C
        assert any(c in result for c in ["A", "B", "C", "category"])

    def test_value_counts_no_df(self):
        set_dataframe(None)
        result = get_value_counts.func("")
        assert "No dataframe" in result


# ── Visualiser tests ──────────────────────────────────────────────────────────

class TestVisualiser:
    def test_generate_charts_returns_list(self, sample_df):
        from src.visualiser import generate_all_charts
        paths = generate_all_charts(sample_df)
        assert isinstance(paths, list)

    def test_charts_are_png_files(self, sample_df):
        from src.visualiser import generate_all_charts
        paths = generate_all_charts(sample_df)
        for path in paths:
            assert path.endswith(".png")
            assert os.path.exists(path)

    def test_missing_value_chart_skipped_for_clean_df(self, sample_df):
        from src.visualiser import plot_missing_values
        # sample_df has no nulls → should return empty list
        result = plot_missing_values(sample_df)
        assert result == []

    def test_missing_value_chart_appears_for_nulls(self, df_with_nulls):
        from src.visualiser import plot_missing_values
        result = plot_missing_values(df_with_nulls)
        assert len(result) >= 1

    def test_correlation_heatmap_needs_two_columns(self):
        from src.visualiser import plot_correlation_heatmap
        df_single = pd.DataFrame({"a": [1, 2, 3]})
        assert plot_correlation_heatmap(df_single) == []

    def test_correlation_heatmap_with_valid_data(self, sample_df):
        from src.visualiser import plot_correlation_heatmap
        result = plot_correlation_heatmap(sample_df)
        assert len(result) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
