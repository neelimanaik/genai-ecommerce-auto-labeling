import json

SYSTEM_PROMPT = """
You are a product text classification system used for e-commerce cataloging.

Task:
Classify a product description into exactly ONE of the following categories:
    - Hair Care
    - Skin Care
    
Instructions:
    - Choose the single most appropriate category.
    - Respond with ONLY the category name.
    - Do not add explanations, symbols, or extra text.
"""
def sanitize(text):
    blocked_terms = ["massage", "body", "waist", "lip", "balm",
        "beard", "scrub", "brush", "mustache", "tattoo", "foot", "nail", "makeup"]
    for term in blocked_terms:
        text = text.replace(term, f"{term} (product)")
    return text

def user_prompt(description: str):
    return {
        "role": "user",
        "content": sanitize(description)
    }

def build_zero_shot_prompt(description: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_prompt(description)
    ]

def build_few_shot_prompt(examples_json: str, description: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in json.loads(examples_json):
        messages.append(user_prompt(ex["Product Description"]))
        messages.append(
            {"role": "assistant", "content": ex["Category"]}
        )
    messages.append(user_prompt(description))
    return messages