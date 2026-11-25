"""
Visualiser
Automatically generates relevant charts based on dataframe column types.
Returns file paths to saved PNG images for display in Gradio gallery.
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List

warnings.filterwarnings("ignore")

OUTPUT_DIR   = "./charts"
STYLE        = "seaborn-v0_8-whitegrid"
FIGSIZE      = (9, 5)
COLOR_MAIN   = "#0066cc"
COLOR_ACCENT = "#ff6600"
MAX_CHARTS   = 8           # cap to avoid overwhelming the gallery


def _save_fig(fig: plt.Figure, name: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_numeric_distributions(df: pd.DataFrame) -> List[str]:
    """Histogram + KDE for each numeric column (up to 6 columns)."""
    paths    = []
    num_cols = df.select_dtypes(include="number").columns[:6].tolist()
    if not num_cols:
        return paths

    cols  = min(3, len(num_cols))
    rows  = (len(num_cols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes  = axes.flatten() if hasattr(axes, "flatten") else [axes]

    with plt.style.context(STYLE):
        for i, col in enumerate(num_cols):
            ax = axes[i]
            df[col].dropna().plot.hist(
                ax=ax, bins=30, color=COLOR_MAIN, edgecolor="white", alpha=0.85
            )
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("Numeric Distributions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    paths.append(_save_fig(fig, "numeric_distributions"))
    return paths


def plot_correlation_heatmap(df: pd.DataFrame) -> List[str]:
    """Correlation heatmap for numeric columns."""
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return []

    try:
        import numpy as np
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)

        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.5 else "black")

        ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return [_save_fig(fig, "correlation_heatmap")]
    except Exception:
        return []


def plot_categorical_bars(df: pd.DataFrame) -> List[str]:
    """Bar charts for categorical columns with 2–20 unique values."""
    paths   = []
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if 2 <= df[c].nunique() <= 20
    ][:3]    # max 3 categorical charts

    for col in cat_cols:
        counts = df[col].value_counts().head(15)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        with plt.style.context(STYLE):
            bars = ax.barh(counts.index[::-1], counts.values[::-1],
                           color=COLOR_MAIN, edgecolor="white", alpha=0.9)
            ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
            ax.set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Count")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        fig.tight_layout()
        paths.append(_save_fig(fig, f"bar_{col.replace(' ', '_')}"))

    return paths


def plot_time_series(df: pd.DataFrame) -> List[str]:
    """Line chart for datetime index or date columns paired with numeric columns."""
    paths    = []
    date_col = None

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break

    if date_col is None:
        return paths

    num_cols = df.select_dtypes(include="number").columns[:3].tolist()
    if not num_cols:
        return paths

    fig, ax = plt.subplots(figsize=(10, 4))
    with plt.style.context(STYLE):
        for i, col in enumerate(num_cols):
            color = [COLOR_MAIN, COLOR_ACCENT, "#2ecc71"][i % 3]
            df.sort_values(date_col).plot(
                x=date_col, y=col, ax=ax, label=col, color=color, linewidth=1.8
            )
        ax.set_title("Time Series", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    paths.append(_save_fig(fig, "time_series"))
    return paths


def plot_missing_values(df: pd.DataFrame) -> List[str]:
    """Bar chart showing null percentage per column (only if nulls exist)."""
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]

    if null_pct.empty:
        return []

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3, len(null_pct) * 0.4)))
    with plt.style.context(STYLE):
        bars = ax.barh(null_pct.index[::-1], null_pct.values[::-1],
                       color="#e74c3c", edgecolor="white", alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return [_save_fig(fig, "missing_values")]


def generate_all_charts(df: pd.DataFrame) -> List[str]:
    """
    Auto-generate all relevant charts for the dataframe.

    Returns:
        List of file paths to saved PNG images
    """
    # Clean up old charts
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(OUTPUT_DIR, f))

    all_paths = []
    generators = [
        plot_numeric_distributions,
        plot_correlation_heatmap,
        plot_categorical_bars,
        plot_time_series,
        plot_missing_values,
    ]

    for gen in generators:
        try:
            paths = gen(df)
            all_paths.extend(paths)
            if len(all_paths) >= MAX_CHARTS:
                break
        except Exception as e:
            print(f"Chart generation warning ({gen.__name__}): {e}")
            continue

    return all_paths[:MAX_CHARTS]
