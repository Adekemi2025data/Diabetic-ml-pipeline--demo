# src/nl_pipeline.py
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.rag_engine import retrieve_context
from src.llm_client import call_llm

PARSER_SYSTEM = """
You are a parser. Extract intent, features, and missing info.
Return ONLY valid JSON.
"""

ANSWER_SYSTEM = """
You are a medical assistant. Use retrieved context + user input.
Be safe, factual, and helpful.
"""

def parse_user_input(user_prompt: str):
    payload = {
        "system": PARSER_SYSTEM,
        "user": user_prompt
    }
    raw = call_llm(payload)
    try:
        return json.loads(raw)
    except:
        return {"intent": "unknown", "raw": raw}

def generate_answer(user_prompt: str, context: str):
    payload = {
        "system": ANSWER_SYSTEM,
        "context": context,
        "user": user_prompt
    }
    return call_llm(payload)

def handle_query(query: str):
    parsed = parse_user_input(query)
    context = retrieve_context(query)
    answer = generate_answer(query, context)
    return answer
