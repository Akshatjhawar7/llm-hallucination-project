import time
import requests
from utils.config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from openai import OpenAI

def query_ollama(prompt):
    start = time.time()
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 256,
        }


    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    end = time.time()

    return {
        "response": data.get("response", "").strip(),
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency": end - start
    }

def query_openai(prompt):

    client = OpenAI(api_key=OPENAI_API_KEY)
    start = time.time()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    end = time.time()

    return {
        "response": response.choices[0].message.content.strip(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency": end - start
    }

def query_llm(prompt):
    if LLM_PROVIDER.lower() == "ollama":
        return query_ollama(prompt)
    elif LLM_PROVIDER.lower() == "openai":
        return query_openai(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")