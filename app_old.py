import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from evaluator import run_evaluation, MODELS
from prompts import PROMPT_CATEGORIES
import json
import time
import os
from datetime import datetime
from pathlib import Path

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

/* Insights box */
.insights-box {
    background: linear-gradient(135deg, #0d2b1a 0%, #0a1a1e 100%);
    border: 1px solid #00ff8840;
    border-left: 3px solid #00ff88;
    border-radius: 4px;
    padding: 16px;
    margin: 12px 0;
    font-size: 0.85rem;
    line-height: 1.8;
    color: #99bb99;
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

/* Leaderboard */
.leaderboard-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: #0d0d1a;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    margin: 8px 0;
    font-size: 0.9rem;
}

.leaderboard-rank {
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #ff9944;
    font-size: 1.4rem;
    min-width: 40px;
}

.leaderboard-medal { font-size: 1.3rem; margin-right: 8px; }

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


# ── Utility functions ─────────────────────────────────────────────────────────
def get_leaderboard_path():
    """Get the path to the leaderboard CSV."""
    return Path("leaderboard.csv")


def load_leaderboard():
    """Load accumulated leaderboard data."""
    path = get_leaderboard_path()
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_to_leaderboard(results_df):
    """Append results to leaderboard CSV."""
    leaderboard = load_leaderboard()
    
    # Prepare results for leaderboard (aggregated by model)
    model_stats = results_df.groupby("model").agg({
        "instruction_following": ["mean", "std"],
        "factual_accuracy": ["mean", "std"],
        "conciseness": ["mean", "std"],
        "naturalness": ["mean", "std"],
        "format_adherence": ["mean", "std"],
        "overall": ["mean", "std"]
    }).round(2)
    
    for model in model_stats.index:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": MODELS[model]["label"],
            "model_key": model,
            "num_evals": len(results_df[results_df["model"] == model]),
            "overall_mean": model_stats.loc[model, ("overall", "mean")],
            "overall_std": model_stats.loc[model, ("overall", "std")],
            "instruction_following_mean": model_stats.loc[model, ("instruction_following", "mean")],
            "factual_accuracy_mean": model_stats.loc[model, ("factual_accuracy", "mean")],
            "conciseness_mean": model_stats.loc[model, ("conciseness", "mean")],
            "naturalness_mean": model_stats.loc[model, ("naturalness", "mean")],
            "format_adherence_mean": model_stats.loc[model, ("format_adherence", "mean")],
        }
        leaderboard = pd.concat([leaderboard, pd.DataFrame([entry])], ignore_index=True)
    
    leaderboard.to_csv(get_leaderboard_path(), index=False)
    return leaderboard


def generate_insights(df, selected_models):
    """Generate AI-interpreted findings from results."""
    insights = []
    
    model_avg = df.groupby("model")[["instruction_following", "factual_accuracy",
                                      "conciseness", "naturalness", "format_adherence"]].mean()
    
    # Find best/worst performers per dimension
    dims = ["instruction_following", "factual_accuracy", "conciseness", "naturalness", "format_adherence"]
    dim_names = ["Instruction Following", "Factual Accuracy", "Conciseness", "Naturalness", "Format Adherence"]
    
    # Overall winner
    best_overall = model_avg["instruction_following"].mean() + model_avg["factual_accuracy"].mean() + \
                   model_avg["conciseness"].mean() + model_avg["naturalness"].mean() + \
                   model_avg["format_adherence"].mean()
    best_model = model_avg.sum(axis=1).idxmax()
    best_label = MODELS[best_model]["label"]
    overall_score = round((model_avg.sum(axis=1).max() / 5), 2)
    insights.append(f"🏆 {best_label} achieves the highest overall score at {overall_score}/10, demonstrating balanced performance across all evaluation dimensions.")
    
    # Category-level insights
    cat_perf = df.groupby(["model", "category"])["overall"].mean().unstack()
    if len(cat_perf.columns) > 1:
        best_reasoning = cat_perf.loc[:, "reasoning"].idxmax() if "reasoning" in cat_perf.columns else best_model
        best_creative = cat_perf.loc[:, "creative_writing"].idxmax() if "creative_writing" in cat_perf.columns else best_model
        insights.append(f"📊 Category specialists: {MODELS[best_reasoning]['label']} excels at reasoning tasks, while {MODELS[best_creative]['label']} delivers superior creative writing output.")
    
    # Variance analysis (confidence)
    std_by_model = df.groupby("model")["overall"].std()
    most_consistent = std_by_model.idxmin()
    insights.append(f"🎯 Consistency: {MODELS[most_consistent]['label']} shows the most stable performance (σ={std_by_model[most_consistent]:.2f}), indicating reliable output quality across diverse prompts.")
    
    # Dimension strengths
    for model in selected_models:
        if model in model_avg.index:
            strong_dims = [dim_names[i] for i, v in enumerate(model_avg.loc[model, dims]) if v >= 8.0]
            weak_dims = [dim_names[i] for i, v in enumerate(model_avg.loc[model, dims]) if v <= 6.0]
            if strong_dims:
                insights.append(f"✓ {MODELS[model]['label']} demonstrates strength in {' and '.join(strong_dims)} tasks.")
    
    return insights[:4]  # Return top 4 insights


def calculate_variance_by_model(df):
    """Calculate variance metrics for confidence intervals."""
    variance_data = []
    
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        overall_scores = model_data["overall"].values
        
        variance_data.append({
            "model": MODELS[model]["label"],
            "mean": overall_scores.mean(),
            "std": overall_scores.std(),
            "min": overall_scores.min(),
            "max": overall_scores.max(),
            "ci_95_lower": overall_scores.mean() - 1.96 * overall_scores.std() / (len(overall_scores) ** 0.5),
            "ci_95_upper": overall_scores.mean() + 1.96 * overall_scores.std() / (len(overall_scores) ** 0.5),
        })
    
    return pd.DataFrame(variance_data)


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
    num_runs = st.slider("Runs per prompt (for confidence intervals)", 1, 3, 1)

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
    · Format Adherence<br>
    <br>
    RUN {num_runs}x for variance analysis<br>
    All results saved to leaderboard
    </div>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "eval_complete" not in st.session_state:
    st.session_state.eval_complete = False


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# LLM Evaluation Dashboard")
st.markdown(
    '<p style="color: #444460; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; margin-top: -12px;">Professional benchmarking · Groq API · LLM-as-Judge · Confidence Intervals</p>',
    unsafe_allow_html=True
)

# ── Navigation ────────────────────────────────────────────────────────────────
nav_tab1, nav_tab2, nav_tab3, nav_tab4 = st.tabs(["📊 Current Run", "📈 Leaderboard", "🎯 Insights", "⚙️ Custom Eval"])

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
                num_runs=num_runs,
                progress_callback=lambda p, msg: (
                    progress_bar.progress(min(p, 0.99)),
                    status_text.markdown(f'<p style="color: #666680; font-size: 0.8rem;">{msg}</p>', unsafe_allow_html=True)
                )
            )

            st.session_state.results = results
            st.session_state.eval_complete = True
            progress_bar.empty()
            status_text.empty()
            
            # Save to leaderboard
            results_df = pd.DataFrame(results)
            save_to_leaderboard(results_df)
            
            st.success("✓ Evaluation complete and saved to leaderboard")


# ── TAB 1: Current Run Results ────────────────────────────────────────────────
with nav_tab1:
    if st.session_state.results:
        results = st.session_state.results
        df = pd.DataFrame(results)

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)

        # Aggregate by model (average across runs)
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
            total_evals = len(df)
            st.markdown(f"""
            <div class="metric-card orange">
                <div class="metric-value" style="color: #ff6b35;">{total_evals}</div>
                <div class="metric-label">Total Evaluations</div>
                <div style="margin-top: 8px; font-size: 0.72rem; color: #888;">{len(df) // len(selected_models)} prompts × {len(selected_models)} models × {num_runs} runs</div>
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

        # ── Variance Analysis (Confidence Intervals) ──────────────────────────
        if num_runs > 1:
            st.markdown('<div class="section-header">Confidence Intervals (95%)</div>', unsafe_allow_html=True)
            variance_df = calculate_variance_by_model(df)
            
            fig_var = go.Figure()
            for _, row in variance_df.iterrows():
                colors = {"Llama 3.1 8B": "#00ff88", "Mixtral 8x7B": "#ff9944", "Gemma 2 9B": "#4fc3f7"}
                color = colors.get(row["model"], "#ffffff")
                
                fig_var.add_trace(go.Scatter(
                    x=[row["ci_95_lower"], row["mean"], row["ci_95_upper"]],
                    y=[row["model"], row["model"], row["model"]],
                    mode='lines+markers',
                    name=row["model"],
                    line=dict(color=color, width=2),
                    marker=dict(size=10, color=color),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[row["mean"] - row["ci_95_lower"]],
                        arrayminus=[row["ci_95_upper"] - row["mean"]]
                    )
                ))
            
            fig_var.update_layout(
                paper_bgcolor="#0a0a0f",
                plot_bgcolor="#0d0d1a",
                font=dict(color="#e8e8e0"),
                xaxis_title="Overall Score ± 95% CI",
                xaxis=dict(gridcolor="#1e1e2e", range=[0, 10.5]),
                yaxis=dict(gridcolor="#1e1e2e"),
                height=300,
                showlegend=False,
                margin=dict(l=150, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_var, use_container_width=True)
            
            st.markdown("""
            <p style="font-size: 0.75rem; color: #666680; font-style: italic;">
            Confidence intervals show score variance across multiple runs. Narrower intervals indicate more consistent performance.
            </p>
            """, unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Radar & Scores", "🗂️ Category Breakdown", "🔬 Response Comparison", "📋 Raw Data"])

        colors = {"llama-3.1-8b-instant": "#00ff88",
                  "mixtral-8x7b-32768": "#ff9944",
                  "gemma2-9b-it": "#4fc3f7"}

        with tab1:
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown('<div class="section-header">Radar — All Dimensions</div>', unsafe_allow_html=True)

                dimensions = ["instruction_following", "factual_accuracy", "conciseness",
                              "naturalness", "format_adherence"]
                dim_labels = ["Instruction\nFollowing", "Factual\nAccuracy",
                              "Conciseness", "Naturalness", "Format\nAdherence"]

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
            st.markdown('<div class="section-header">Performance by Category</div>', unsafe_allow_html=True)

            cat_breakdown = df.groupby(["category", "model"])["overall"].agg(["mean", "std", "count"]).reset_index()
            
            categories = df["category"].unique()
            for category in sorted(categories):
                cat_data = cat_breakdown[cat_breakdown["category"] == category]
                
                fig_cat = go.Figure()
                for _, row in cat_data.iterrows():
                    model_key = row["model"]
                    color = colors.get(model_key, "#ffffff")
                    fig_cat.add_trace(go.Bar(
                        name=MODELS[model_key]["label"],
                        x=[MODELS[model_key]["label"]],
                        y=[row["mean"]],
                        error_y=dict(type='data', array=[row["std"]]),
                        marker=dict(color=color, opacity=0.85),
                        text=f"{row['mean']:.2f}",
                        textposition="outside"
                    ))
                
                fig_cat.update_layout(
                    title=f"📌 {category.upper().replace('_', ' ')}",
                    paper_bgcolor="#0a0a0f",
                    plot_bgcolor="#0d0d1a",
                    font=dict(color="#e8e8e0", size=10),
                    xaxis=dict(gridcolor="#1e1e2e"),
                    yaxis=dict(range=[0, 10.5], gridcolor="#1e1e2e"),
                    height=250,
                    showlegend=False,
                    margin=dict(l=40, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_cat, use_container_width=True, key=f"cat_{category}")

        with tab3:
            st.markdown('<div class="section-header">Response Comparison by Prompt</div>', unsafe_allow_html=True)

            prompts_list = df["prompt"].unique().tolist()
            selected_prompt_idx = st.selectbox(
                "Select prompt",
                options=range(len(prompts_list)),
                format_func=lambda i: f"[{df[df['prompt']==prompts_list[i]]['category'].iloc[0].upper()}] {prompts_list[i][:80]}..."
                if len(prompts_list[i]) > 80 else f"[{df[df['prompt']==prompts_list[i]]['category'].iloc[0].upper()}] {prompts_list[i]}"
            )

            selected_prompt = prompts_list[selected_prompt_idx]
            prompt_df = df[df["prompt"] == selected_prompt].drop_duplicates(subset=["model"])

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

        with tab4:
            st.markdown('<div class="section-header">Raw Evaluation Data</div>', unsafe_allow_html=True)

            display_cols = ["model", "category", "prompt", "instruction_following",
                            "factual_accuracy", "conciseness", "naturalness", "format_adherence", "overall", "run_num"]
            display_df = df[display_cols].copy()
            display_df["model"] = display_df["model"].map(lambda x: MODELS[x]["label"])
            display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

            st.dataframe(display_df, use_container_width=True, height=400)

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("⚡ Run an evaluation to see results here.")


# ── TAB 2: Leaderboard ────────────────────────────────────────────────────────
with nav_tab2:
    st.markdown('<div class="section-header">Accumulated Leaderboard</div>', unsafe_allow_html=True)
    
    leaderboard = load_leaderboard()
    
    if not leaderboard.empty:
        # Aggregate total stats
        leaderboard_agg = leaderboard.groupby("model").agg({
            "overall_mean": "mean",
            "num_evals": "sum",
            "instruction_following_mean": "mean",
            "factual_accuracy_mean": "mean",
            "conciseness_mean": "mean",
            "naturalness_mean": "mean",
            "format_adherence_mean": "mean"
        }).round(2).sort_values("overall_mean", ascending=False)
        
        leaderboard_agg["rank"] = range(1, len(leaderboard_agg) + 1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            for idx, (model, row) in enumerate(leaderboard_agg.iterrows()):
                medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "•"
                
                st.markdown(f"""
                <div class="leaderboard-row">
                    <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
                        <span class="leaderboard-medal">{medal}</span>
                        <span class="leaderboard-rank">#{int(row['rank'])}</span>
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #e8e8e0;">{model}</div>
                            <div style="font-size: 0.7rem; color: #666680;">{int(row['num_evals'])} evaluations</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2rem; font-weight: 800; color: #00ff88; font-family: 'JetBrains Mono';">{row['overall_mean']:.2f}</div>
                        <div style="font-size: 0.7rem; color: #666680;">overall score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-header">Dimension Averages</div>', unsafe_allow_html=True)
            
            dim_avg = leaderboard_agg[[
                "instruction_following_mean",
                "factual_accuracy_mean",
                "conciseness_mean",
                "naturalness_mean",
                "format_adherence_mean"
            ]].mean()
            
            for dim, val in dim_avg.items():
                dim_name = dim.replace("_mean", "").replace("_", " ").title()
                st.metric(dim_name, f"{val:.2f}/10")
        
        st.markdown("---")
        st.markdown('<div class="section-header">Leaderboard History</div>', unsafe_allow_html=True)
        
        # Show recent entries
        recent = leaderboard.tail(10).sort_values("timestamp", ascending=False)
        display_recent = recent[["timestamp", "model", "overall_mean", "overall_std", "num_evals"]].copy()
        display_recent.columns = ["Timestamp", "Model", "Score", "Variance", "Evals"]
        st.dataframe(display_recent, use_container_width=True)
        
        # Download leaderboard
        csv_leaderboard = leaderboard.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Leaderboard",
            data=csv_leaderboard,
            file_name=f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📊 No leaderboard data yet. Run evaluations to populate the leaderboard.")


# ── TAB 3: Insights ───────────────────────────────────────────────────────────
with nav_tab3:
    st.markdown('<div class="section-header">AI-Generated Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        insights = generate_insights(df, selected_models)
        
        for insight in insights:
            st.markdown(f'<div class="insights-box">{insight}</div>', unsafe_allow_html=True)
        
        # Additional analysis
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        
        model_avg = df.groupby("model")[["instruction_following", "factual_accuracy",
                                          "conciseness", "naturalness", "format_adherence"]].mean()
        
        comparison = []
        for model in selected_models:
            if model in model_avg.index:
                comparison.append({
                    "Model": MODELS[model]["label"],
                    "Avg Score": round(model_avg.loc[model].mean(), 2),
                    "Instruction": round(model_avg.loc[model, "instruction_following"], 2),
                    "Accuracy": round(model_avg.loc[model, "factual_accuracy"], 2),
                    "Conciseness": round(model_avg.loc[model, "conciseness"], 2),
                    "Naturalness": round(model_avg.loc[model, "naturalness"], 2),
                    "Format": round(model_avg.loc[model, "format_adherence"], 2),
                })
        
        comp_df = pd.DataFrame(comparison)
        st.dataframe(comp_df, use_container_width=True)
        
    else:
        st.info("🎯 Run an evaluation to generate insights.")


# ── TAB 4: Custom Evaluation ──────────────────────────────────────────────────
with nav_tab4:
    st.markdown('<div class="section-header">Custom Prompt Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 0.85rem; color: #888899; line-height: 1.6;">
    Enter your own prompt and evaluate it live across selected models. Great for testing custom prompts and understanding model strengths in your specific use case.
    </p>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        custom_prompt = st.text_area(
            "Your custom prompt",
            placeholder="E.g., 'Write a professional email about...'",
            height=150,
            label_visibility="collapsed"
        )
    
    with col_right:
        custom_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        custom_models = st.multiselect(
            "Models to test",
            options=list(MODELS.keys()),
            default=list(MODELS.keys())[:1],
            format_func=lambda x: MODELS[x]["label"],
            label_visibility="collapsed"
        )
        custom_judge = st.selectbox(
            "Judge model",
            options=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            label_visibility="collapsed"
        )
        run_custom = st.button("▶  EVALUATE CUSTOM PROMPT", use_container_width=True)
    
    if run_custom:
        if not custom_prompt.strip():
            st.error("⚠ Please enter a prompt to evaluate.")
        elif not custom_api_key:
            st.error("⚠ Please provide your Groq API key.")
        elif not custom_models:
            st.error("⚠ Select at least one model.")
        else:
            from evaluator import get_model_response, judge_response
            from groq import Groq
            
            client = Groq(api_key=custom_api_key)
            
            with st.spinner("Evaluating custom prompt..."):
                custom_results = []
                
                for model in custom_models:
                    progress_text = st.empty()
                    progress_text.text(f"Getting response from {MODELS[model]['label']}...")
                    
                    response = get_model_response(client, model, custom_prompt)
                    time.sleep(0.3)
                    
                    progress_text.text(f"Judging {MODELS[model]['label']} response...")
                    scores = judge_response(client, custom_judge, custom_prompt, response)
                    
                    score_keys = ["instruction_following", "factual_accuracy", "conciseness", "naturalness", "format_adherence"]
                    overall = sum(scores.get(k, 5) for k in score_keys) / len(score_keys)
                    
                    custom_results.append({
                        "model": model,
                        "response": response,
                        "scores": scores,
                        "overall": overall
                    })
                    
                    time.sleep(0.5)
                
                progress_text.empty()
                st.success("✓ Evaluation complete")
            
            st.markdown("---")
            st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
            
            cols = st.columns(len(custom_results))
            for i, result in enumerate(custom_results):
                with cols[i]:
                    model_key = result["model"]
                    color = colors.get(model_key, "#ffffff")
                    
                    st.markdown(f"""
                    <div style='border-top: 2px solid {color}; padding-top: 12px; margin-bottom: 8px;'>
                        <div style='font-size: 0.8rem; font-weight: 700; color: {color};'>{MODELS[model_key]['label']}</div>
                        <div style='font-size: 1.1rem; font-weight: 800; color: {color}; font-family: JetBrains Mono;'>{result['overall']:.1f}/10</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="response-box">{result["response"]}</div>', unsafe_allow_html=True)
                    
                    judge_text = result["scores"].get("reasoning", "No feedback available.")
                    st.markdown(f'<div class="judge-verdict">🧑‍⚖️ {judge_text}</div>', unsafe_allow_html=True)
                    
                    for dim, label in [("instruction_following", "Instr"), ("naturalness", "Natural"),
                                       ("conciseness", "Concise"), ("factual_accuracy", "Factual")]:
                        score = result["scores"].get(dim, 5)
                        pct = score * 10
                        st.markdown(f"""
                        <div class="score-bar-container">
                            <div class="score-bar-label"><span>{label}</span><span style="color: {color};">{score:.1f}</span></div>
                            <div class="score-bar"><div class="score-fill" style="width: {pct}%; background: {color};"></div></div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insights-box" style="margin-top: 32px;">
        💡 <strong>Use cases:</strong> Test domain-specific prompts, compare model quality on your exact use case, or explore new prompt engineering techniques interactively.
        </div>
        """, unsafe_allow_html=True)
