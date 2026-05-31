import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

class OpenAIClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 60) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
