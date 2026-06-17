from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
m_key = os.getenv("MISTRAL_API_KEY")
g_key = os.getenv("GEMINI_API_KEY")

print(f"MISTRAL_API_KEY: [{m_key}]")
print(f"GEMINI_API_KEY: [{g_key}]")
