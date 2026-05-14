import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from evaluator import run_evaluation, MODELS
from prompts import PROMPT_CATEGORIES
import json
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* Dark industrial theme */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8e0;
}

[data-testid="stSidebar"] {
    background-color: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #141428 100%);
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    padding: 20px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #00ff88;
}
.metric-card.orange::before { background: #ff6b35; }
.metric-card.blue::before { background: #4fc3f7; }
.metric-card.purple::before { background: #b39ddb; }

.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    color: #00ff88;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #666680;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #444460;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Model tag pills */
.model-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.tag-llama { background: #0d2b1a; color: #00ff88; border: 1px solid #00ff8840; }
.tag-mixtral { background: #2b1a0d; color: #ff9944; border: 1px solid #ff994440; }
.tag-gemma { background: #0d1a2b; color: #4fc3f7; border: 1px solid #4fc3f740; }

/* Score bar */
.score-bar-container { margin: 4px 0; }
.score-bar-label {
    font-size: 0.68rem;
    color: #888899;
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}
.score-bar {
    height: 4px;
    border-radius: 2px;
    background: #1e1e2e;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
}

/* Response comparison box */
.response-box {
    background: #0d0d1a;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 16px;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #c8c8d0;
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
    max-height: 280px;
    overflow-y: auto;
}

.judge-verdict {
    background: #0a1a0d;
    border: 1px solid #00ff8830;
    border-left: 3px solid #00ff88;
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #99bb99;
    margin-top: 8px;
}

/* Streamlit overrides */
.stButton > button {
    background: #00ff88 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
}
.stButton > button:hover {
    background: #00cc6a !important;
    transform: translateY(-1px);
}

.stSelectbox label, .stSlider label, .stTextInput label {
    color: #666680 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
}

h1 { 
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: #e8e8e0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ LLM EVAL")
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown('<div class="section-header">Models</div>', unsafe_allow_html=True)
    selected_models = st.multiselect(
        "Select models to compare",
        options=list(MODELS.keys()),
        default=list(MODELS.keys()),
        format_func=lambda x: MODELS[x]["label"]
    )

    st.markdown('<div class="section-header">Evaluation</div>', unsafe_allow_html=True)
    selected_categories = st.multiselect(
        "Prompt categories",
        options=list(PROMPT_CATEGORIES.keys()),
        default=list(PROMPT_CATEGORIES.keys())
    )

    num_prompts = st.slider("Prompts per category", 1, 5, 2)

    st.markdown('<div class="section-header">Judge Model</div>', unsafe_allow_html=True)
    judge_model = st.selectbox(
        "LLM-as-judge",
        options=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )

    st.markdown("---")
    run_eval = st.button("▶  RUN EVALUATION", use_container_width=True)

    st.markdown("""
    <div style='margin-top: 32px; font-size: 0.65rem; color: #333350; line-height: 1.8;'>
    EVAL DIMENSIONS<br>
    · Instruction Following<br>
    · Factual Accuracy<br>
    · Response Conciseness<br>
    · Natural Language Quality<br>
    · Format Adherence
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# LLM Evaluation Dashboard")
st.markdown(
    '<p style="color: #444460; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; margin-top: -12px;">Open-source model benchmarking · Groq API · LLM-as-Judge</p>',
    unsafe_allow_html=True
)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "eval_complete" not in st.session_state:
    st.session_state.eval_complete = False


# ── Run evaluation ────────────────────────────────────────────────────────────
if run_eval:
    if not api_key:
        st.error("⚠ Enter your Groq API key in the sidebar.")
    elif not selected_models:
        st.error("⚠ Select at least one model.")
    elif not selected_categories:
        st.error("⚠ Select at least one prompt category.")
    else:
        with st.spinner("Running evaluation..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = run_evaluation(
                api_key=api_key,
                models=selected_models,
                categories=selected_categories,
                num_prompts=num_prompts,
                judge_model=judge_model,
                progress_callback=lambda p, msg: (
                    progress_bar.progress(p),
                    status_text.markdown(f'<p style="color: #666680; font-size: 0.8rem;">{msg}</p>', unsafe_allow_html=True)
                )
            )

            st.session_state.results = results
            st.session_state.eval_complete = True
            progress_bar.empty()
            status_text.empty()
            st.success("✓ Evaluation complete")


# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    df = pd.DataFrame(results)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)

    model_avg = df.groupby("model")[["instruction_following", "factual_accuracy",
                                      "conciseness", "naturalness", "format_adherence"]].mean()
    model_avg["overall"] = model_avg.mean(axis=1)

    best_model_key = model_avg["overall"].idxmax()
    best_model_label = MODELS[best_model_key]["label"]
    best_score = model_avg.loc[best_model_key, "overall"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_score:.1f}</div>
            <div class="metric-label">Top Overall Score</div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #00ff88;">{best_model_label}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        total_prompts = len(df) // len(selected_models)
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="metric-value" style="color: #ff6b35;">{len(df)}</div>
            <div class="metric-label">Total Evaluations</div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #888;">{total_prompts} prompts × {len(selected_models)} models</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        best_instr = model_avg["instruction_following"].idxmax()
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-value" style="color: #4fc3f7;">{model_avg.loc[best_instr, 'instruction_following']:.1f}</div>
            <div class="metric-label">Best Instruction Following</div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #4fc3f7;">{MODELS[best_instr]['label']}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        best_nat = model_avg["naturalness"].idxmax()
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-value" style="color: #b39ddb;">{model_avg.loc[best_nat, 'naturalness']:.1f}</div>
            <div class="metric-label">Most Natural Output</div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #b39ddb;">{MODELS[best_nat]['label']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  Radar & Scores", "🔬  Response Comparison", "📋  Raw Results"])

    with tab1:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<div class="section-header">Radar — All Dimensions</div>', unsafe_allow_html=True)

            dimensions = ["instruction_following", "factual_accuracy", "conciseness",
                          "naturalness", "format_adherence"]
            dim_labels = ["Instruction\nFollowing", "Factual\nAccuracy",
                          "Conciseness", "Naturalness", "Format\nAdherence"]

            colors = {"llama-3.1-8b-instant": "#00ff88",
                      "mixtral-8x7b-32768": "#ff9944",
                      "gemma2-9b-it": "#4fc3f7"}

            fig = go.Figure()
            for model_key in selected_models:
                if model_key in model_avg.index:
                    vals = model_avg.loc[model_key, dimensions].tolist()
                    vals_closed = vals + [vals[0]]
                    labels_closed = dim_labels + [dim_labels[0]]
                    color = colors.get(model_key, "#ffffff")
                    fig.add_trace(go.Scatterpolar(
                        r=vals_closed,
                        theta=labels_closed,
                        fill='toself',
                        fillcolor=color + "22",
                        line=dict(color=color, width=2),
                        name=MODELS[model_key]["label"]
                    ))

            fig.update_layout(
                polar=dict(
                    bgcolor="#0d0d1a",
                    radialaxis=dict(visible=True, range=[0, 10],
                                   gridcolor="#1e1e2e", tickcolor="#444460",
                                   tickfont=dict(color="#444460", size=9)),
                    angularaxis=dict(gridcolor="#1e1e2e", tickcolor="#888899",
                                    tickfont=dict(color="#888899", size=10))
                ),
                paper_bgcolor="#0a0a0f",
                plot_bgcolor="#0a0a0f",
                font=dict(color="#e8e8e0"),
                legend=dict(bgcolor="#0f0f18", bordercolor="#1e1e2e", borderwidth=1,
                            font=dict(size=11)),
                margin=dict(l=40, r=40, t=20, b=20),
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">Overall Score by Model</div>', unsafe_allow_html=True)

            bar_colors = [colors.get(m, "#ffffff") for m in model_avg.index]
            fig2 = go.Figure(go.Bar(
                x=[MODELS[m]["label"] for m in model_avg.index],
                y=model_avg["overall"].values,
                marker=dict(color=bar_colors, opacity=0.85),
                text=[f"{v:.2f}" for v in model_avg["overall"].values],
                textposition="outside",
                textfont=dict(color="#e8e8e0", size=13, family="JetBrains Mono")
            ))
            fig2.update_layout(
                paper_bgcolor="#0a0a0f",
                plot_bgcolor="#0d0d1a",
                font=dict(color="#e8e8e0"),
                xaxis=dict(gridcolor="#1e1e2e", tickfont=dict(size=11)),
                yaxis=dict(range=[0, 10.5], gridcolor="#1e1e2e", tickfont=dict(size=10)),
                margin=dict(l=20, r=20, t=20, b=20),
                height=200,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)

            dim_short = {
                "instruction_following": "Instruction Following",
                "factual_accuracy": "Factual Accuracy",
                "conciseness": "Conciseness",
                "naturalness": "Naturalness",
                "format_adherence": "Format Adherence"
            }
            score_data = []
            for model_key in selected_models:
                if model_key in model_avg.index:
                    for dim, label in dim_short.items():
                        score_data.append({
                            "Model": MODELS[model_key]["label"],
                            "Dimension": label,
                            "Score": round(model_avg.loc[model_key, dim], 2)
                        })

            score_df = pd.DataFrame(score_data)
            pivot = score_df.pivot(index="Dimension", columns="Model", values="Score")
            st.dataframe(
                pivot.style.background_gradient(cmap="Greens", vmin=0, vmax=10)
                     .format("{:.2f}"),
                use_container_width=True
            )

    with tab2:
        st.markdown('<div class="section-header">Response Comparison by Prompt</div>', unsafe_allow_html=True)

        # Group by prompt
        prompts_list = df["prompt"].unique().tolist()
        selected_prompt_idx = st.selectbox(
            "Select prompt",
            options=range(len(prompts_list)),
            format_func=lambda i: f"[{df[df['prompt']==prompts_list[i]]['category'].iloc[0].upper()}] {prompts_list[i][:80]}..."
            if len(prompts_list[i]) > 80 else f"[{df[df['prompt']==prompts_list[i]]['category'].iloc[0].upper()}] {prompts_list[i]}"
        )

        selected_prompt = prompts_list[selected_prompt_idx]
        prompt_df = df[df["prompt"] == selected_prompt]

        st.markdown(f"""
        <div style='background: #0d0d1a; border: 1px solid #2a2a4a; border-radius: 4px; padding: 14px 18px; margin: 12px 0;'>
            <div style='font-size: 0.65rem; color: #444460; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 6px;'>Prompt</div>
            <div style='font-size: 0.9rem; color: #c8c8d0; line-height: 1.6;'>{selected_prompt}</div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(prompt_df))
        for i, (_, row) in enumerate(prompt_df.iterrows()):
            with cols[i]:
                color = colors.get(row["model"], "#ffffff")
                st.markdown(f"""
                <div style='border-top: 2px solid {color}; padding-top: 12px; margin-bottom: 8px;'>
                    <div style='font-size: 0.8rem; font-weight: 700; color: {color};'>{MODELS[row['model']]['label']}</div>
                    <div style='font-size: 0.65rem; color: #444460; font-family: JetBrains Mono;'>overall: {row.get('overall', 0):.1f}/10</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="response-box">{row["response"]}</div>', unsafe_allow_html=True)

                judge_text = row.get("judge_reasoning", "No judge feedback available.")
                st.markdown(f'<div class="judge-verdict">🧑‍⚖️ {judge_text}</div>', unsafe_allow_html=True)

                # Mini score breakdown
                for dim, label in [("instruction_following", "Instr"), ("naturalness", "Natural"),
                                   ("conciseness", "Concise"), ("factual_accuracy", "Factual")]:
                    score = row.get(dim, 0)
                    pct = score * 10
                    st.markdown(f"""
                    <div class="score-bar-container">
                        <div class="score-bar-label"><span>{label}</span><span style="color: {color};">{score:.1f}</span></div>
                        <div class="score-bar"><div class="score-fill" style="width: {pct}%; background: {color};"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">Raw Evaluation Data</div>', unsafe_allow_html=True)

        display_cols = ["model", "category", "prompt", "instruction_following",
                        "factual_accuracy", "conciseness", "naturalness", "format_adherence", "overall"]
        display_df = df[display_cols].copy()
        display_df["model"] = display_df["model"].map(lambda x: MODELS[x]["label"])
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

        st.dataframe(display_df, use_container_width=True, height=400)

        csv = display_df.to_csv(index=False)
        st.download_button(
            "⬇  Download CSV",
            data=csv,
            file_name="llm_eval_results.csv",
            mime="text/csv"
        )

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align: center; padding: 80px 40px; color: #222235;'>
        <div style='font-size: 4rem; margin-bottom: 16px;'>⚡</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #333350; letter-spacing: 0.1em; text-transform: uppercase;'>
            Configure & Run Evaluation
        </div>
        <div style='font-size: 0.8rem; color: #222235; margin-top: 8px;'>
            Add your Groq API key → select models → click Run
        </div>
    </div>
    """, unsafe_allow_html=True)
