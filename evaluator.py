"""
evaluator.py — Core evaluation logic
Uses Groq API to:
  1. Get responses from candidate models
  2. Score responses using a judge model (LLM-as-judge)
"""

import json
import time
from groq import Groq
from prompts import PROMPT_CATEGORIES

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "llama-3.1-8b-instant": {
        "label": "Llama 3.1 8B",
        "color": "#00ff88"
    },
    "mixtral-8x7b-32768": {
        "label": "Mixtral 8x7B",
        "color": "#ff9944"
    },
    "gemma2-9b-it": {
        "label": "Gemma 2 9B",
        "color": "#4fc3f7"
    }
}

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert LLM evaluator. You will be given a prompt and a model's response.
Score the response on 5 dimensions, each from 1 to 10.

Definitions:
- instruction_following: Did the model do exactly what was asked? Did it follow format/length/tone constraints?
- factual_accuracy: Is the content factually correct and free of hallucinations?
- conciseness: Is the response appropriately concise — no unnecessary padding, repetition, or verbosity?
- naturalness: Does the response sound natural and human-like, not robotic or AI-generated?
- format_adherence: Did the response match any requested format (bullet points, list, JSON, etc.)?

Return ONLY valid JSON, no preamble, no markdown:
{
  "instruction_following": <int 1-10>,
  "factual_accuracy": <int 1-10>,
  "conciseness": <int 1-10>,
  "naturalness": <int 1-10>,
  "format_adherence": <int 1-10>,
  "reasoning": "<one concise sentence explaining the key strength or weakness>"
}"""


def get_model_response(client: Groq, model: str, prompt: str) -> str:
    """Get a response from a candidate model."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Follow instructions precisely."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def judge_response(client: Groq, judge_model: str, prompt: str, response: str) -> dict:
    """Score a response using the judge model."""
    judge_prompt = f"""PROMPT GIVEN TO MODEL:
{prompt}

MODEL RESPONSE:
{response}

Evaluate the response on the 5 dimensions and return JSON only."""

    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt}
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = completion.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
        return scores
    except Exception as e:
        # Fallback if judge fails
        return {
            "instruction_following": 5,
            "factual_accuracy": 5,
            "conciseness": 5,
            "naturalness": 5,
            "format_adherence": 5,
            "reasoning": f"[Judge error: {str(e)}]"
        }


def run_evaluation(
    api_key: str,
    models: list,
    categories: list,
    num_prompts: int,
    judge_model: str,
    progress_callback=None
) -> list:
    """
    Main evaluation runner.
    Returns a list of result dicts, one per (model, prompt) pair.
    """
    client = Groq(api_key=api_key)

    # Build prompt list
    prompts_to_run = []
    for category in categories:
        category_prompts = PROMPT_CATEGORIES.get(category, [])[:num_prompts]
        for prompt in category_prompts:
            prompts_to_run.append({"category": category, "prompt": prompt})

    total_steps = len(prompts_to_run) * len(models)
    step = 0
    results = []

    for prompt_item in prompts_to_run:
        category = prompt_item["category"]
        prompt = prompt_item["prompt"]

        for model in models:
            step += 1
            progress = step / total_steps

            if progress_callback:
                progress_callback(
                    progress,
                    f"[{step}/{total_steps}] Running {MODELS[model]['label']} on {category} prompt..."
                )

            # Get model response
            response = get_model_response(client, model, prompt)

            # Small delay to respect rate limits
            time.sleep(0.3)

            # Judge the response
            if progress_callback:
                progress_callback(
                    progress,
                    f"[{step}/{total_steps}] Judging {MODELS[model]['label']} response..."
                )

            scores = judge_response(client, judge_model, prompt, response)

            # Compute overall
            score_keys = ["instruction_following", "factual_accuracy",
                          "conciseness", "naturalness", "format_adherence"]
            overall = sum(scores.get(k, 5) for k in score_keys) / len(score_keys)

            results.append({
                "model": model,
                "category": category,
                "prompt": prompt,
                "response": response,
                "instruction_following": scores.get("instruction_following", 5),
                "factual_accuracy": scores.get("factual_accuracy", 5),
                "conciseness": scores.get("conciseness", 5),
                "naturalness": scores.get("naturalness", 5),
                "format_adherence": scores.get("format_adherence", 5),
                "overall": round(overall, 2),
                "judge_reasoning": scores.get("reasoning", "")
            })

            # Delay between models on same prompt
            time.sleep(0.5)

    return results
