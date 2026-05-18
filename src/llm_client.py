import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEBIUS_API_KEY")
ENDPOINT = os.getenv("NEBIUS_ENDPOINT")
MODEL = os.getenv("NEBIUS_MODEL")

def call_llm(messages):
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "input": messages[-1]["content"],
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(ENDPOINT, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"LLM Error {response.status_code}: {response.text}")

    return response.json()["output_text"]

