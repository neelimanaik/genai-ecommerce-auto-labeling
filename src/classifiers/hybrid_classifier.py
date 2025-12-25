from src.llm.client import call_llm
from src.llm.prompts import  SYSTEM_PROMPT, build_zero_shot_prompt, build_few_shot_prompt, user_prompt
from src.rules.rule_classifier import rule_based_classify
from src.classifiers.llm_classifier import classify_zero_shot, classify_few_shot

SAFE_FALLBACK = "Manual Review"   # or "Hair Care" / "Manual Review"
def classify_hybrid(description: str, examples_json=None):
    label, confidence = rule_based_classify(description)

    if confidence == "CONFIDENT":
        return label

    
    # fallback to LLM
    if examples_json:
        messages = build_few_shot_prompt(examples_json, description)
        result = call_llm(messages)
        if result is None:
            return SAFE_FALLBACK
    else:
        messages = build_zero_shot_prompt(description)
        result = call_llm(messages)
        if result is None:
            return SAFE_FALLBACK
    
    return result
