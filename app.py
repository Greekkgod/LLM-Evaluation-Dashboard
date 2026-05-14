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
    page_title="Model Benchmark Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Enhanced Professional CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

.stApp { background: linear-gradient(135deg, #f8f9fa 0%, #eef2f5 100%); color: #1a1f36; }
[data-testid="stSidebar"] { display: none; }

/* Hero/Header */
.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 48px 32px;
    border-radius: 12px;
    color: white;
    margin-bottom: 32px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
}
.hero-title { font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1.2; }
.hero-subtitle { font-size: 1.1rem; opacity: 0.95; margin-top: 8px; font-weight: 300; }
.hero-desc { font-size: 0.95rem; opacity: 0.85; margin-top: 16px; max-width: 600px; line-height: 1.6; }

/* Wizard */
.wizard-container {
    background: white;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    border: 1px solid #e0e4f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.wizard-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
}
.wizard-step-indicator {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #667eea;
    font-weight: 700;
}
.wizard-step-number {
    display: inline-block;
    width: 32px;
    height: 32px;
    background: #667eea;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    font-weight: 700;
    font-size: 0.9rem;
}

/* Model cards */
.model-option {
    background: white;
    border: 2px solid #e0e4f0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.2s;
}
.model-option:hover {
    border-color: #667eea;
    background: #f5f7ff;
    transform: translateX(4px);
}
.model-option.selected {
    border-color: #667eea;
    background: #f5f7ff;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
}
.model-name { font-weight: 600; color: #1a1f36; font-size: 1rem; }
.model-desc { font-size: 0.85rem; color: #6b7280; margin-top: 4px; }

/* Progress */
.progress-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0;
    font-size: 0.95rem;
    border-bottom: 1px solid #f0f4f8;
}
.progress-item:last-child { border-bottom: none; }
.progress-icon { font-size: 1.2rem; }
.progress-text { flex: 1; }
.progress-status { font-size: 0.85rem; color: #9ca3af; }

/* Insights */
.insight-card {
    background: linear-gradient(135deg, #f5f7ff 0%, #fafbff 100%);
    border: 1px solid #e0e4f0;
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #374151;
}
.insight-card strong { color: #1a1f36; font-weight: 600; }

/* Leaderboard */
.leaderboard-container {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e0e4f0;
    margin: 16px 0;
}
.leaderboard-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0;
    border-bottom: 1px solid #f0f4f8;
}
.leaderboard-row:last-child { border-bottom: none; }
.leaderboard-rank {
    font-size: 1.4rem;
    font-weight: 800;
    color: #667eea;
    min-width: 40px;
}
.leaderboard-medal { font-size: 1.3rem; margin-right: 4px; }
.leaderboard-info {
    flex: 1;
    margin-left: 16px;
}
.leaderboard-model { font-weight: 600; color: #1a1f36; }
.leaderboard-evals { font-size: 0.85rem; color: #9ca3af; }
.leaderboard-score {
    font-size: 1.3rem;
    font-weight: 800;
    color: #667eea;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 16px 0;
}
.metric-card {
    background: white;
    border: 1px solid #e0e4f0;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #667eea; }
.metric-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #9ca3af; margin-top: 4px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3) !important;
}

/* Form elements */
.stSelectbox label, .stSlider label, .stTextInput label {
    color: #1a1f36 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

h1 { color: #1a1f36 !important; font-weight: 800 !important; }
h2 { color: #1a1f36 !important; font-weight: 700 !important; margin-top: 24px !important; }
h3 { color: #374151 !important; font-weight: 600 !important; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #9ca3af;
}
.empty-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-title { font-size: 1.2rem; font-weight: 600; color: #1a1f36; margin: 12px 0; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Utility functions ─────────────────────────────────────────────────────────
def get_leaderboard_path():
    return Path("leaderboard.csv")

def load_leaderboard():
    path = get_leaderboard_path()
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def save_to_leaderboard(results_df):
    leaderboard = load_leaderboard()
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
        }
        leaderboard = pd.concat([leaderboard, pd.DataFrame([entry])], ignore_index=True)
    
    leaderboard.to_csv(get_leaderboard_path(), index=False)
    return leaderboard

def plain_language_insight(results_df, selected_models):
    """Generate human-readable insights."""
    insights = []
    
    model_avg = results_df.groupby("model")[["instruction_following", "factual_accuracy",
                                              "conciseness", "naturalness", "format_adherence"]].mean()
    model_avg["overall"] = model_avg.mean(axis=1)
    
    best_model = model_avg["overall"].idxmax()
    best_score = model_avg.loc[best_model, "overall"]
    best_label = MODELS[best_model]["label"]
    
    insights.append(f"🏆 **{best_label} wins overall** with a score of {best_score:.1f}/10. It's the most well-rounded performer across all dimensions.")
    
    # Find strengths
    for model in selected_models:
        if model in model_avg.index:
            scores = model_avg.loc[model]
            best_dim = scores.idxmax()
            best_dim_score = scores[best_dim]
            if best_dim != "overall" and best_dim_score >= 8.0:
                dim_name = best_dim.replace("_", " ").title()
                insights.append(f"💡 **{MODELS[model]['label']}** excels at {dim_name} ({best_dim_score:.1f}/10)—use it for tasks requiring this skill.")
    
    # Consistency
    std_by_model = results_df.groupby("model")["overall"].std()
    most_consistent = std_by_model.idxmin()
    if std_by_model[most_consistent] < 1.5:
        insights.append(f"🎯 **{MODELS[most_consistent]['label']}** is the most consistent performer. Its quality doesn't vary much between different prompts.")
    
    return insights[:3]

def generate_progress_updates(step, total_steps, current_model, current_category, current_phase):
    """Generate human-friendly progress messages."""
    progress = step / total_steps
    percent = int(progress * 100)
    
    messages = []
    model_label = MODELS[current_model]["label"]
    
    if current_phase == "response":
        messages.append(f"🤖 Asking {model_label} about {current_category}...")
        messages.append(f"⏳ This takes a moment...")
    elif current_phase == "judge":
        messages.append(f"🧑‍⚖️ Judge evaluating the response...")
        messages.append(f"📊 Scoring on 5 dimensions...")
    
    messages.append(f"📈 Progress: {percent}% ({step}/{total_steps})")
    
    return messages, progress

# ── Session State ─────────────────────────────────────────────────────────────
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0  # 0=welcome, 1=models, 2=categories, 3=config, 4=ready
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_complete" not in st.session_state:
    st.session_state.eval_complete = False
if "wizard_state" not in st.session_state:
    st.session_state.wizard_state = {
        "models": list(MODELS.keys()),
        "categories": list(PROMPT_CATEGORIES.keys()),
        "num_prompts": 2,
        "num_runs": 1,
        "judge_model": "llama-3.1-8b-instant",
        "api_key": ""
    }

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🔬 Model Benchmark Studio</div>
    <div class="hero-subtitle">Find the perfect model for your use case</div>
    <div class="hero-desc">Benchmark open-source LLMs on real tasks. Compare reasoning, creativity, accuracy, and more. All results saved to your persistent leaderboard.</div>
</div>
""", unsafe_allow_html=True)

# ── Get Leaderboard for quick stats ───────────────────────────────────────────
leaderboard = load_leaderboard()

if not leaderboard.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        total_evals = leaderboard["num_evals"].sum()
        st.metric("Total Evaluations Run", int(total_evals))
    with col2:
        unique_models = leaderboard["model"].nunique()
        st.metric("Models Tested", int(unique_models))
    with col3:
        best_overall = leaderboard.groupby("model")["overall_mean"].mean().max()
        st.metric("Highest Score Achieved", f"{best_overall:.2f}/10")

st.divider()

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_eval, tab_leaderboard, tab_insights, tab_custom = st.tabs(
    ["🚀 Run Evaluation", "🏆 Leaderboard", "💡 Insights", "⚙️ Custom Test"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: RUN EVALUATION (Setup Wizard)
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    
    # Welcome screen
    if st.session_state.wizard_step == 0:
        st.markdown("""
        ### Welcome! Let's benchmark some models.
        
        This wizard will guide you through 3 simple steps:
        1. **Choose models** to test
        2. **Pick what to test them on** (reasoning, creativity, etc.)
        3. **Run and compare**
        
        Ready to see which model is best for your needs? Let's go! 👇
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Benchmark", use_container_width=True, key="start_wizard"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Skip to Leaderboard", use_container_width=True):
                st.switch_page("app.py?page=leaderboard")
    
    # Step 1: Choose Models
    elif st.session_state.wizard_step == 1:
        st.markdown("""
        <div class="wizard-container">
            <div class="wizard-header">
                <div class="wizard-step-number">1</div>
                <div>
                    <div class="wizard-step-indicator">Step 1 of 3</div>
                    <h3 style="margin:0;">Which models do you want to benchmark?</h3>
                </div>
            </div>
            <p style="color: #6b7280; margin-bottom: 20px;">Select one or more models. We'll run them through the same tests and show you how they compare.</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected = st.multiselect(
            "Pick your models",
            options=list(MODELS.keys()),
            default=st.session_state.wizard_state["models"],
            format_func=lambda x: MODELS[x]["label"],
            label_visibility="collapsed"
        )
        
        if selected:
            st.session_state.wizard_state["models"] = selected
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back", use_container_width=True):
                    st.session_state.wizard_step = 0
                    st.rerun()
            with col2:
                if st.button("Next →", use_container_width=True):
                    st.session_state.wizard_step = 2
                    st.rerun()
        else:
            st.warning("Please select at least one model")
    
    # Step 2: Choose Categories
    elif st.session_state.wizard_step == 2:
        st.markdown("""
        <div class="wizard-container">
            <div class="wizard-header">
                <div class="wizard-step-number">2</div>
                <div>
                    <div class="wizard-step-indicator">Step 2 of 3</div>
                    <h3 style="margin:0;">What should we test them on?</h3>
                </div>
            </div>
            <p style="color: #6b7280; margin-bottom: 20px;">Choose one or more evaluation categories. Models will be tested on prompts from each category.</p>
        </div>
        """, unsafe_allow_html=True)
        
        category_descriptions = {
            "instruction_following": "Can the model follow strict rules and constraints?",
            "factual_accuracy": "Does it know facts and avoid making things up?",
            "creative_writing": "Is it creative, natural, and engaging?",
            "reasoning": "Can it think through complex problems?",
            "summarization": "Can it extract and compress information?"
        }
        
        for cat in PROMPT_CATEGORIES.keys():
            st.checkbox(
                f"**{cat.replace('_', ' ').title()}** — {category_descriptions.get(cat, '')}",
                value=cat in st.session_state.wizard_state["categories"],
                key=f"cat_{cat}",
                on_change=lambda: None
            )
        
        # Update state from checkboxes
        selected_cats = [cat for cat in PROMPT_CATEGORIES.keys() if st.session_state.get(f"cat_{cat}")]
        if selected_cats:
            st.session_state.wizard_state["categories"] = selected_cats
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.session_state.wizard_state["categories"]:
                if st.button("Next →", use_container_width=True):
                    st.session_state.wizard_step = 3
                    st.rerun()
            else:
                st.warning("Please select at least one category")
    
    # Step 3: Fine-tune Settings
    elif st.session_state.wizard_step == 3:
        st.markdown("""
        <div class="wizard-container">
            <div class="wizard-header">
                <div class="wizard-step-number">3</div>
                <div>
                    <div class="wizard-step-indicator">Step 3 of 3</div>
                    <h3 style="margin:0;">Let's fine-tune your benchmark</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "🔑 Groq API Key (get free at console.groq.com)",
                type="password",
                placeholder="gsk_...",
                value=st.session_state.wizard_state.get("api_key", ""),
                help="Your API key stays secure and is never saved."
            )
            st.session_state.wizard_state["api_key"] = api_key
        
        with col2:
            num_prompts = st.slider(
                "📝 How many prompts per category?",
                min_value=1,
                max_value=5,
                value=st.session_state.wizard_state["num_prompts"],
                help="More = more thorough, but takes longer"
            )
            st.session_state.wizard_state["num_prompts"] = num_prompts
        
        num_runs = st.slider(
            "🔄 Run each prompt N times (for variance analysis)",
            min_value=1,
            max_value=3,
            value=st.session_state.wizard_state["num_runs"],
            help="Multiple runs show consistency. 1x is quick, 3x is thorough."
        )
        st.session_state.wizard_state["num_runs"] = num_runs
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if api_key:
                if st.button("🚀 Start Benchmark", use_container_width=True, type="primary"):
                    st.session_state.wizard_step = 4
                    st.rerun()
            else:
                st.error("⚠️ Please enter your Groq API key")
    
    # Step 4: Running Evaluation
    elif st.session_state.wizard_step == 4:
        st.markdown("""
        <div class="wizard-container">
            <div class="wizard-header">
                <div class="wizard-step-number">🔄</div>
                <div>
                    <div class="wizard-step-indicator">Running Benchmark</div>
                    <h3 style="margin:0;">Your benchmark is running</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        progress_log = progress_container.empty()
        
        try:
            all_results = []
            total_evals = (len(st.session_state.wizard_state["models"]) * 
                          len(st.session_state.wizard_state["categories"]) * 
                          st.session_state.wizard_state["num_prompts"] * 
                          st.session_state.wizard_state["num_runs"])
            
            eval_count = 0
            messages_log = []
            
            def progress_callback(progress, message):
                nonlocal eval_count, messages_log
                eval_count += 1
                messages_log.append(message)
                
                # Keep last 8 messages
                if len(messages_log) > 8:
                    messages_log = messages_log[-8:]
                
                progress_bar.progress(min(progress, 0.99))
                with progress_log:
                    st.markdown("**Recent activity:**")
                    for msg in messages_log:
                        st.caption(f"✓ {msg}")
            
            results = run_evaluation(
                api_key=st.session_state.wizard_state["api_key"],
                models=st.session_state.wizard_state["models"],
                categories=st.session_state.wizard_state["categories"],
                num_prompts=st.session_state.wizard_state["num_prompts"],
                judge_model=st.session_state.wizard_state["judge_model"],
                num_runs=st.session_state.wizard_state["num_runs"],
                progress_callback=progress_callback
            )
            
            st.session_state.eval_results = results
            st.session_state.eval_complete = True
            
            # Save to leaderboard
            results_df = pd.DataFrame(results)
            save_to_leaderboard(results_df)
            
            progress_bar.progress(1.0)
            progress_log.empty()
            
            st.success("✅ Benchmark complete! Your results have been saved to the leaderboard.")
            st.session_state.wizard_step = 5
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Oops! Something went wrong: {str(e)}")
            st.info("💡 **Troubleshooting:**\n- Check your API key is valid\n- Make sure you have API credits\n- Try with fewer prompts or runs")
            if st.button("← Try Again"):
                st.session_state.wizard_step = 3
                st.rerun()
    
    # Step 5: Results Display
    elif st.session_state.wizard_step == 5 and st.session_state.eval_results:
        results = st.session_state.eval_results
        df = pd.DataFrame(results)
        
        st.markdown("### 🎯 Your Benchmark Results")
        
        # Aggregate scores
        model_avg = df.groupby("model")[["instruction_following", "factual_accuracy",
                                          "conciseness", "naturalness", "format_adherence"]].mean()
        model_avg["overall"] = model_avg.mean(axis=1)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        best_model = model_avg["overall"].idxmax()
        with col1:
            st.metric("🏆 Top Performer", MODELS[best_model]["label"], 
                     f"{model_avg.loc[best_model, 'overall']:.2f}/10")
        with col2:
            st.metric("📊 Total Evals", len(df))
        with col3:
            best_instr = model_avg["instruction_following"].idxmax()
            st.metric("🎯 Best at Following Orders", MODELS[best_instr]["label"])
        with col4:
            best_nat = model_avg["naturalness"].idxmax()
            st.metric("🗣️ Most Natural", MODELS[best_nat]["label"])
        
        # Plain language insights
        st.markdown("### 💡 Key Findings")
        insights = plain_language_insight(df, st.session_state.wizard_state["models"])
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        # Detailed charts
        with st.expander("📊 See detailed breakdown"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                dimensions = ["instruction_following", "factual_accuracy", "conciseness",
                            "naturalness", "format_adherence"]
                dim_labels = ["Instruction", "Accuracy", "Conciseness", "Naturalness", "Format"]
                
                fig = go.Figure()
                colors = {"llama-3.1-8b-instant": "#667eea",
                         "mixtral-8x7b-32768": "#f59e0b",
                         "gemma2-9b-it": "#10b981"}
                
                for model_key in st.session_state.wizard_state["models"]:
                    if model_key in model_avg.index:
                        vals = model_avg.loc[model_key, dimensions].tolist()
                        vals_closed = vals + [vals[0]]
                        labels_closed = dim_labels + [dim_labels[0]]
                        fig.add_trace(go.Scatterpolar(
                            r=vals_closed, theta=labels_closed,
                            fill='toself', fillcolor=colors.get(model_key, "#667eea") + "33",
                            line=dict(color=colors.get(model_key, "#667eea"), width=2),
                            name=MODELS[model_key]["label"]
                        ))
                
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 10])
                    ),
                    paper_bgcolor="white",
                    font=dict(color="#1a1f36"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score comparison
                bar_colors = [colors.get(m, "#667eea") for m in model_avg.index]
                fig2 = go.Figure(go.Bar(
                    x=[MODELS[m]["label"] for m in model_avg.index],
                    y=model_avg["overall"].values,
                    marker=dict(color=bar_colors),
                    text=[f"{v:.2f}" for v in model_avg["overall"].values],
                    textposition="outside"
                ))
                fig2.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#1a1f36"),
                    yaxis=dict(range=[0, 10.5]),
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Download results
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Run Another Benchmark"):
                st.session_state.wizard_step = 0
                st.rerun()
        with col2:
            if st.button("📊 Go to Leaderboard"):
                st.switch_page("app.py?tab=leaderboard")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: LEADERBOARD (Historical data)
# ════════════════════════════════════════════════════════════════════════════
with tab_leaderboard:
    st.markdown("### 🏆 All-Time Model Rankings")
    
    if not leaderboard.empty:
        # Aggregate stats
        leaderboard_agg = leaderboard.groupby("model").agg({
            "overall_mean": ["mean", "count"],
            "num_evals": "sum"
        }).round(2)
        leaderboard_agg.columns = ["avg_score", "num_tests", "total_evals"]
        leaderboard_agg = leaderboard_agg.sort_values("avg_score", ascending=False)
        leaderboard_agg["rank"] = range(1, len(leaderboard_agg) + 1)
        
        st.markdown("""
        Your persistent leaderboard accumulates all benchmarks over time. The more you test, the clearer the picture.
        """)
        
        for idx, (model, row) in enumerate(leaderboard_agg.iterrows()):
            medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "•"
            
            st.markdown(f"""
            <div class="leaderboard-container">
                <div class="leaderboard-row">
                    <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
                        <span class="leaderboard-medal">{medal}</span>
                        <span class="leaderboard-rank">#{int(row['rank'])}</span>
                        <div class="leaderboard-info">
                            <div class="leaderboard-model">{model}</div>
                            <div class="leaderboard-evals">{int(row['total_evals'])} evaluations • {int(row['num_tests'])} test runs</div>
                        </div>
                    </div>
                    <div class="leaderboard-score">{row['avg_score']:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Benchmarks Run", len(leaderboard))
        with col2:
            st.metric("Models Tested", leaderboard["model"].nunique())
        with col3:
            st.metric("All-Time High Score", f"{leaderboard['overall_mean'].max():.2f}/10")
        
        # Download leaderboard
        st.download_button(
            label="📥 Export Leaderboard",
            data=leaderboard.to_csv(index=False),
            file_name=f"leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">No benchmarks yet</div>
            <p>Run your first benchmark to start building the leaderboard!</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Run First Benchmark", use_container_width=True):
            st.session_state.wizard_step = 0
            st.switch_page("app.py?tab=eval")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("### 💡 Data-Driven Insights")
    
    if st.session_state.eval_results:
        df = pd.DataFrame(st.session_state.eval_results)
        
        st.markdown("""
        Based on your most recent benchmark, here's what we learned about your models:
        """)
        
        insights = plain_language_insight(df, st.session_state.wizard_state["models"])
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        # Category breakdown
        st.markdown("### 📚 Performance by Category")
        
        cat_perf = df.groupby(["category", "model"])["overall"].mean().unstack()
        
        for category in cat_perf.index:
            best_for_cat = cat_perf.loc[category].idxmax()
            best_score = cat_perf.loc[category].max()
            st.info(f"**{category.replace('_', ' ').title()}**: {best_for_cat} leads with {best_score:.1f}/10")
    
    elif not leaderboard.empty:
        st.markdown("### Historical Trends")
        st.info("💡 Run a fresh benchmark to see insights about your current models.")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <div class="empty-title">No insights yet</div>
            <p>Run a benchmark first to generate insights</p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: CUSTOM TEST
# ════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown("""
    ### ⚙️ Test Your Own Prompt
    
    Have a specific use case? Write your own prompt and see how each model handles it.
    """)
    
    custom_prompt = st.text_area(
        "Your prompt",
        placeholder="E.g., 'Write a professional email requesting a meeting about Q3 results'",
        height=120,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        custom_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        custom_models = st.multiselect(
            "Models to test",
            options=list(MODELS.keys()),
            default=list(MODELS.keys())[:1],
            format_func=lambda x: MODELS[x]["label"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Preview:**")
        if custom_prompt:
            st.caption(custom_prompt[:100] + "..." if len(custom_prompt) > 100 else custom_prompt)
    
    if st.button("🚀 Evaluate Now", use_container_width=True):
        if not custom_prompt.strip():
            st.error("⚠️ Please enter a prompt")
        elif not custom_api_key:
            st.error("⚠️ Please provide your Groq API key")
        elif not custom_models:
            st.error("⚠️ Select at least one model")
        else:
            from evaluator import get_model_response, judge_response
            from groq import Groq
            
            client = Groq(api_key=custom_api_key)
            
            with st.spinner("Testing your prompt..."):
                try:
                    results_container = st.container()
                    
                    for model in custom_models:
                        response = get_model_response(client, model, custom_prompt)
                        time.sleep(0.3)
                        
                        scores = judge_response(client, "llama-3.1-8b-instant", custom_prompt, response)
                        
                        score_keys = ["instruction_following", "factual_accuracy", "conciseness", "naturalness", "format_adherence"]
                        overall = sum(scores.get(k, 5) for k in score_keys) / len(score_keys)
                        
                        with results_container.container():
                            st.markdown(f"#### {MODELS[model]['label']} — {overall:.1f}/10")
                            st.write(response)
                            st.caption(f"🧑‍⚖️ Judge: {scores.get('reasoning', 'No feedback')}")
                        
                        time.sleep(0.5)
                    
                    st.success("✅ Done!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
