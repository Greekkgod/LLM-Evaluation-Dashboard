# ⚡ LLM Evaluation Dashboard

A portfolio project that evaluates open-source LLMs on real quality dimensions using the **LLM-as-Judge** methodology — the same approach used in professional post-training evaluation pipelines.

## What it measures

| Dimension | What it tests |
|---|---|
| **Instruction Following** | Does the model do exactly what was asked? |
| **Factual Accuracy** | Is the output factually correct / hallucination-free? |
| **Conciseness** | Is the response appropriately brief without padding? |
| **Naturalness** | Does it sound human vs robotic / AI-generated? |
| **Format Adherence** | Did it match requested format (bullets, JSON, etc.)? |

## Models compared (via Groq API — free tier)
- `llama-3.1-8b-instant` — Meta's Llama 3.1
- `mixtral-8x7b-32768` — Mistral's MoE model
- `gemma2-9b-it` — Google's Gemma 2

## Methodology: LLM-as-Judge

Instead of manual labeling, a separate LLM (judge model) evaluates each response against a structured rubric. This mirrors industry-standard evaluation approaches (e.g., MT-Bench, AlpacaEval).

The judge scores each response 1–10 per dimension and provides a reasoning explanation. This makes results interpretable, not just numeric.

## Setup

### 1. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) → Sign up → API Keys → Create key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. Use the dashboard
1. Paste your Groq API key in the sidebar
2. Select models to compare (default: all 3)
3. Select prompt categories
4. Set prompts per category (2 is fast, 5 is thorough)
5. Click **Run Evaluation**

## Project structure
```
llm_eval_dashboard/
├── app.py          # Streamlit UI and visualizations
├── evaluator.py    # Groq API calls + LLM-as-judge logic
├── prompts.py      # 25 evaluation prompts across 5 categories
├── requirements.txt
└── README.md
```

## How to talk about this in interviews

**"How do you measure hallucination?"**
> "I use LLM-as-judge where a separate model scores factual accuracy on a 1-10 rubric with explicit reasoning. This is more scalable than human annotation and catches both factual errors and overconfident wrong answers."

**"Why these dimensions?"**
> "Instruction following and naturalness are the two dimensions that most directly predict real-world usefulness — a model that ignores constraints or sounds robotic fails regardless of raw capability."

**"What did you find?"**
> Run the eval and have real numbers ready. That's the point of this project.
