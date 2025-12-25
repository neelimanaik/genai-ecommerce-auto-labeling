# src/classifier.py
from src.llm.client import call_llm
from src.llm.prompts import  SYSTEM_PROMPT, build_zero_shot_prompt, build_few_shot_prompt, user_prompt


def classify_zero_shot(description: str):
    messages = build_zero_shot_prompt(description)
    return call_llm(messages)

def classify_few_shot(description: str, examples_json: str):
    messages = build_few_shot_prompt(examples_json, description)
    return call_llm(messages)