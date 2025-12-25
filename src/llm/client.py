# src/llm_client.py
from openai import AzureOpenAI
from src.config.settings import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_MODEL
)

def get_llm_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

def call_llm(messages, temperature=0, max_tokens=4):
    #print("Calling LLM with messages:",AZURE_OPENAI_MODEL)
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        print("⚠️ LLM blocked or failed:", e)
    return None
    return response.choices[0].message.content.strip()
