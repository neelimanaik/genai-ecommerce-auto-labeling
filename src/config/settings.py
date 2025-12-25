# src/config.py
import os
from dotenv import load_dotenv, find_dotenv

# Try to load `home.env` located next to this file, fall back to any .env found
env_path = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(env_path):
	env_path = find_dotenv()
load_dotenv(env_path)

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

#print("Configuration Loaded (model present):", AZURE_OPENAI_MODEL)