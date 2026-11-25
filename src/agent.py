"""
Agent Core
Builds a LangChain agent with custom pandas tools for data analysis.
Compatible with LangChain 1.2.15 + LangGraph
"""

import io
import traceback
from typing import Any

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Global df reference used by tools
_df: pd.DataFrame = None


def set_dataframe(df: pd.DataFrame) -> None:
    global _df
    _df = df


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_basic_stats(query: str = "") -> str:
    """Returns basic statistics for all numeric columns including count, mean, std, min, max."""
    if _df is None:
        return "No dataframe loaded."
    try:
        return f"Basic statistics:\n{_df.describe().round(3).to_string()}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_column_info(query: str = "") -> str:
    """Returns column names, data types, non-null counts, and null percentages."""
    if _df is None:
        return "No dataframe loaded."
    try:
        buf = io.StringIO()
        _df.info(buf=buf)
        null_pct = (_df.isnull().sum() / len(_df) * 100).round(2)
        return buf.getvalue() + "\nNull % per column:\n" + null_pct.to_string()
    except Exception as e:
        return f"Error: {e}"


@tool
def get_correlation_matrix(query: str = "") -> str:
    """Returns the top 10 strongest correlations between numeric columns."""
    if _df is None:
        return "No dataframe loaded."
    try:
        import numpy as np
        numeric_df = _df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return "Need at least 2 numeric columns."
        corr = numeric_df.corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        pairs = (
            corr.where(mask)
            .stack()
            .reset_index()
        )
        pairs.columns = ["Column A", "Column B", "Correlation"]
        pairs["abs"] = pairs["Correlation"].abs()
        top = pairs.nlargest(10, "abs")[["Column A", "Column B", "Correlation"]]
        return f"Top correlations:\n{top.to_string(index=False)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_outliers(query: str = "") -> str:
    """Detects outliers in numeric columns using the IQR method."""
    if _df is None:
        return "No dataframe loaded."
    try:
        results = []
        for col in _df.select_dtypes(include="number").columns:
            s = _df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = s[(s < lower) | (s > upper)]
            if len(outliers) > 0:
                results.append(
                    f"{col}: {len(outliers)} outliers "
                    f"(bounds [{lower:.2f}, {upper:.2f}]), "
                    f"examples: {outliers.head(3).tolist()}"
                )
        return "Outliers:\n" + "\n".join(results) if results else "No significant outliers detected."
    except Exception as e:
        return f"Error: {e}"


@tool
def get_value_counts(query: str = "") -> str:
    """Returns value counts for categorical columns."""
    if _df is None:
        return "No dataframe loaded."
    try:
        cat_cols = _df.select_dtypes(exclude="number").columns.tolist()
        if not cat_cols:
            return "No categorical columns found."
        output = []
        for col in cat_cols[:5]:
            output.append(f"\n{col}:\n{_df[col].value_counts().head(10).to_string()}")
        return "Value counts:" + "".join(output)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_custom_pandas(code: str) -> str:
    """Executes a custom pandas expression on the dataframe. The dataframe is available as 'df'."""
    if _df is None:
        return "No dataframe loaded."
    try:
        local_vars = {"df": _df, "pd": pd}
        result = eval(code, {"__builtins__": {}}, local_vars)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.head(20).to_string()
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ── Tools list ────────────────────────────────────────────────────────────────

TOOLS = [
    get_basic_stats,
    get_column_info,
    get_correlation_matrix,
    detect_outliers,
    get_value_counts,
    run_custom_pandas,
]

SYSTEM_PROMPT = """You are an expert data analyst. You have access to a pandas DataFrame and analysis tools.

Your job is to:
1. Understand the dataset structure
2. Identify patterns, correlations, outliers, and key statistics
3. Answer the user's question accurately
4. Present findings in clear plain English

Always use multiple tools to build a complete picture. Cite specific numbers. Never make up data."""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(df: pd.DataFrame):
    """Build a LangGraph react agent for pandas data analysis."""
    set_dataframe(df)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2048,
    )
    agent = create_react_agent(
        llm,
        TOOLS,
        prompt=SYSTEM_PROMPT,
    )
    return agent


def run_analysis(agent, df: pd.DataFrame, question: str) -> str:
    """Run the analysis agent against a question."""
    set_dataframe(df)

    context = (
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns. "
        f"Columns: {', '.join(df.columns.tolist())}.\n\nTask: {question}"
    )

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": context}]
        })
        # Extract the last AI message
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                if hasattr(msg, "type") and msg.type == "ai":
                    return msg.content
                elif not hasattr(msg, "type"):
                    return msg.content
        return "No output generated."
    except Exception as e:
        tb = traceback.format_exc()
        return f"Analysis error: {e}\n\n{tb}"