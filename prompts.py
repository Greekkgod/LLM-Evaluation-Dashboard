"""
prompts.py — Evaluation prompt dataset

5 categories that test different model capabilities:
1. instruction_following  — strict format/constraint adherence
2. factual_accuracy       — knowledge and hallucination resistance
3. creative_writing       — naturalness and tone
4. reasoning              — logical thinking
5. summarization          — conciseness and extraction
"""

PROMPT_CATEGORIES = {

    "instruction_following": [
        "List exactly 3 benefits of exercise. Use bullet points. Do not write more than one sentence per bullet.",
        "Write a product description for a coffee mug. Use exactly 50 words. Do not go over or under.",
        "Give me 5 tips for better sleep. Number them 1-5. Each tip must start with a verb.",
        "Explain what a neural network is. Use only words a 10-year-old would understand. Maximum 3 sentences.",
        "Write a haiku about the ocean. Follow the 5-7-5 syllable structure strictly.",
    ],

    "factual_accuracy": [
        "What is the capital of Australia? Give a one-sentence answer.",
        "Who wrote the theory of general relativity and in what year was it published?",
        "What does HTTP stand for and what is it used for?",
        "Name the three laws of motion and who formulated them.",
        "What is the boiling point of water at sea level in both Celsius and Fahrenheit?",
    ],

    "creative_writing": [
        "Write a 3-sentence short story about a robot who discovers music for the first time.",
        "Describe the smell of rain in a forest using only sensory language. Keep it under 60 words.",
        "Write a one-paragraph diary entry from the perspective of a houseplant.",
        "Write a catchy tagline for a fictional company that sells memories.",
        "Describe a busy morning in a city in exactly 4 sentences. Make it vivid.",
    ],

    "reasoning": [
        "If a train travels at 80 km/h and needs to cover 200 km, how long will the journey take? Show your reasoning.",
        "A store sells apples for $2 each and oranges for $3 each. If I spend $18 and buy 3 apples, how many oranges did I buy?",
        "What is the logical flaw in this argument: 'All dogs bark. Max barks. Therefore Max is a dog.'",
        "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? Explain why or why not.",
        "Rank these tasks by priority if your goal is to pass an exam tomorrow: sleeping 8 hours, watching a movie, reviewing notes, eating dinner. Justify your ranking.",
    ],

    "summarization": [
        "Summarize the concept of machine learning in 2 sentences for a business executive with no technical background.",
        "In one sentence, explain what blockchain technology is and why people care about it.",
        "Summarize the main causes of World War 1 in 3 bullet points.",
        "Explain the difference between supervised and unsupervised learning in under 50 words.",
        "Summarize what the internet is and how it works in 3 sentences suitable for a middle school student.",
    ],

}
