"""
Data Loader
Handles CSV loading, validation, and summary generation.
"""

import io
import pandas as pd
from typing import Tuple


SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
MAX_FILE_SIZE_MB    = 50
MAX_ROWS            = 500_000


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Robustly load a CSV file, trying multiple encodings.

    Args:
        filepath : path to the CSV file

    Returns:
        Loaded and lightly cleaned pandas DataFrame

    Raises:
        ValueError : if the file cannot be parsed or is too large
    """
    df = None
    last_error = None

    for enc in SUPPORTED_ENCODINGS:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, Exception) as e:
            last_error = e
            continue

    if df is None:
        raise ValueError(
            f"Could not parse CSV with any supported encoding. "
            f"Last error: {last_error}"
        )

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"Dataset too large ({len(df):,} rows). Maximum supported: {MAX_ROWS:,}."
        )

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = _clean_dataframe(df)
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning: strip whitespace from column names and string values."""
    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Strip string values
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Try to parse date-like columns
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "time", "year", "month"]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except Exception:
                pass

    return df


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a rich text summary of the dataframe for display.

    Args:
        df : pandas DataFrame

    Returns:
        Formatted multi-line string summary
    """
    lines = [
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        "",
        "── Columns ──",
    ]

    for col in df.columns:
        dtype    = str(df[col].dtype)
    nulls    = df[col].isnull().sum()
    null_pct = nulls / len(df) * 100

    if df[col].dtype in ["object", "category"] or str(df[col].dtype) == "string":
        n_unique = df[col].nunique()
        lines.append(f"  {col:<30} {dtype:<12} {nulls} nulls ({null_pct:.1f}%)  |  {n_unique} unique values")
    elif pd.api.types.is_numeric_dtype(df[col]):
        try:
            col_min  = df[col].min()
            col_max  = df[col].max()
            col_mean = df[col].mean()
            lines.append(
                f"  {col:<30} {dtype:<12} {nulls} nulls ({null_pct:.1f}%)"
                f"  |  min={col_min}, max={col_max}, mean={col_mean:.2f}"
            )
        except Exception:
            lines.append(f"  {col:<30} {dtype:<12} {nulls} nulls ({null_pct:.1f}%)")
    else:
        lines.append(f"  {col:<30} {dtype:<12} {nulls} nulls ({null_pct:.1f}%)")
           

    lines += [
        "",
        "── First 5 rows ──",
        df.head(5).to_string(),
        "",
        "── Basic stats (numeric) ──",
        df.describe().round(3).to_string() if not df.select_dtypes(include="number").empty else "No numeric columns.",
    ]

    return "\n".join(lines)
