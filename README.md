# Automated Data Analysis Agent

Upload a CSV file. Ask a question in plain English. Get a full analysis with charts and insights — powered by LangChain + OpenAI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-lightgrey?logo=openai)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## How it works

```
Upload CSV
    |
    v
[Data Loader]          Validates, cleans, parses dates
    |
    v
[Visualiser]           Auto-generates charts (distributions, correlations,
    |                  time series, missing values, bar charts)
    v
[LangChain Agent]      GPT-3.5 + 6 custom pandas tools
    |
    |- get_basic_stats       Descriptive statistics
    |- get_column_info       Schema, dtypes, null counts
    |- get_correlation_matrix  Top correlations
    |- detect_outliers       IQR-based anomaly detection
    |- get_value_counts      Categorical distributions
    |- run_custom_pandas     Flexible ad-hoc queries
    |
    v
Plain-English Insights + Source Charts
```

---

## Features

- Upload any CSV up to 500k rows
- 6 specialised analysis tools the agent chooses from autonomously
- Auto-generated charts: histograms, correlation heatmap, bar charts, time series, missing value map
- Plain-English output suitable for both technical and non-technical audiences
- Ask specific questions or let the agent run a full autonomous analysis
- Clean project structure with unit tests

---

## Project Structure

```
data-analysis-agent/
├── app.py                   # Gradio UI — main entry point
├── src/
│   ├── agent.py             # LangChain agent + 6 pandas tools
│   ├── data_loader.py       # CSV loading, cleaning, summary
│   └── visualiser.py        # Auto chart generation (matplotlib)
├── tests/
│   └── test_agent.py        # Unit tests (pytest)
├── data/
│   └── sample_csvs/
│       └── sales_sample.csv # Sample dataset to test with
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/rubanbuddha/data-analysis-agent.git
cd data-analysis-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your key
```

### 5. Launch
```bash
python app.py
# Open http://localhost:7860
```

### 6. Try it out
Upload `data/sample_csvs/sales_sample.csv` and ask:
- "Which product category generates the most revenue?"
- "Are there any seasonal trends in the sales data?"
- "Which region performs best and why?"

---

## Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Agent Tools

| Tool | What it does |
|---|---|
| `get_basic_stats` | Descriptive stats for all numeric columns |
| `get_column_info` | Schema, dtypes, null counts and percentages |
| `get_correlation_matrix` | Top 10 strongest column correlations |
| `detect_outliers` | IQR-based outlier detection per column |
| `get_value_counts` | Distribution of categorical columns |
| `run_custom_pandas` | Execute ad-hoc pandas expressions |

The agent autonomously decides which tools to call based on the question, then synthesises all results into a single coherent response.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Gradio |
| LLM | OpenAI GPT-3.5-turbo |
| Agent framework | LangChain (OpenAI Tools Agent) |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Testing | Pytest |

---

## Key Concepts Demonstrated

- LangChain agent with custom tool definitions
- Autonomous multi-step reasoning (agent picks tools dynamically)
- Pandas-based data analysis pipeline
- Auto-adaptive chart generation based on data types
- Clean separation of concerns (loader / agent / visualiser / UI)

---

## Author

Md. Hasibur Rahman
chat@hasib.is-a.dev | linkedin.com/in/HasibAlTawha
