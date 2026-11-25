"""
Automated Data Analysis Agent
Author: Ruban Buddha Kanmani Rajan
Description: Upload a CSV → AI agent runs analysis, generates visualisations,
             and returns plain-English insight summaries powered by LangChain + OpenAI.
"""

import os
import gradio as gr
from dotenv import load_dotenv
from src.agent import build_agent, run_analysis
from src.data_loader import load_csv, get_dataframe_summary
from src.visualiser import generate_all_charts

load_dotenv()

# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
#title        { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0; }
#subtitle     { text-align: center; color: #666; margin-top: 0; margin-bottom: 1.5rem; }
.panel        { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; }
.insight-box  { background: #f0f7ff; border-left: 4px solid #0066cc;
                padding: 1rem; border-radius: 4px; white-space: pre-wrap; }
"""


def process_csv(file, user_question: str, api_key: str):
    """
    Main pipeline: load CSV → build agent → run analysis → generate charts.

    Args:
        file          : Gradio file object (path to uploaded CSV)
        user_question : optional specific question from the user
        api_key       : OpenAI API key

    Yields:
        Tuples of (status_msg, summary_text, charts_list, insights_text)
    """
    if file is None:
        yield "Please upload a CSV file.", "", [], "No file uploaded."
        return

    if not api_key.strip():
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.strip():
        yield "Please provide an OpenAI API key.", "", [], ""
        return

    os.environ["OPENAI_API_KEY"] = api_key

    # Step 1: Load & validate CSV
    yield "Loading CSV…", "", [], ""
    try:
        df = load_csv(file.name)
        summary = get_dataframe_summary(df)
    except Exception as e:
        yield f"Error loading file: {e}", "", [], ""
        return

    # Step 2: Generate visualisations
    yield "Generating charts…", summary, [], ""
    try:
        charts = generate_all_charts(df)
    except Exception as e:
        charts = []
        print(f"Chart generation warning: {e}")

    # Step 3: Build agent and run analysis
    yield "Running AI analysis agent…", summary, charts, ""
    try:
        agent   = build_agent(df)
        default = (
            "Perform a comprehensive analysis of this dataset. "
            "Include: (1) key statistics and distributions, "
            "(2) notable correlations or patterns, "
            "(3) outliers or anomalies, "
            "(4) actionable business insights, "
            "(5) recommendations for further analysis."
        )
        question = user_question.strip() if user_question.strip() else default
        insights = run_analysis(agent, df, question)
    except Exception as e:
        insights = f"Agent error: {e}\n\nBasic summary:\n{summary}"

    yield "Analysis complete!", summary, charts, insights


def build_ui():
    with gr.Blocks(css=CSS, title="Data Analysis Agent") as demo:

        gr.Markdown("# Automated Data Analysis Agent", elem_id="title")
        gr.Markdown(
            "Upload a CSV file → the AI agent analyses it and returns "
            "plain-English insights with visualisations.",
            elem_id="subtitle",
        )

        with gr.Row():
            # ── Left panel: inputs ────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                api_key_box = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-… (or set OPENAI_API_KEY in .env)",
                    value=os.getenv("OPENAI_API_KEY", ""),
                )

                file_input = gr.File(
                    label="Upload CSV",
                    file_types=[".csv"],
                )

                question_box = gr.Textbox(
                    label="Specific question (optional)",
                    placeholder="e.g. Which product category has the highest revenue?",
                    lines=3,
                )

                analyse_btn = gr.Button("Analyse Dataset", variant="primary")

                with gr.Accordion("Example questions", open=False):
                    gr.Markdown("""
- What are the top 5 correlations in this dataset?
- Are there any outliers I should investigate?
- Which features are most important for predicting the target variable?
- What trends exist over time?
- Summarise this dataset for a non-technical stakeholder.
                    """)

            # ── Right panel: outputs ──────────────────────────────────────────
            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)

                with gr.Tabs():
                    with gr.Tab("AI Insights"):
                        insights_box = gr.Textbox(
                            label="Analysis",
                            interactive=False,
                            lines=20,
                            elem_classes=["insight-box"],
                        )

                    with gr.Tab("Data Summary"):
                        summary_box = gr.Textbox(
                            label="DataFrame Summary",
                            interactive=False,
                            lines=20,
                        )

                    with gr.Tab("Charts"):
                        gallery = gr.Gallery(
                            label="Visualisations",
                            show_label=False,
                            columns=2,
                            height="auto",
                        )

        analyse_btn.click(
            fn=process_csv,
            inputs=[file_input, question_box, api_key_box],
            outputs=[status_box, summary_box, gallery, insights_box],
        )

        gr.Markdown(
            "Built by [Ruban Buddha](https://linkedin.com/in/rubanbuddha) "
            "· LangChain · OpenAI · Gradio · Pandas · Matplotlib",
            elem_id="subtitle",
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )