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

def query_ollama(prompt, temperature=0):
    start = time.time()
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
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

def query_openai(prompt, temperature=0):

    client = OpenAI(api_key=OPENAI_API_KEY)
    start = time.time()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    end = time.time()

    return {
        "response": response.choices[0].message.content.strip(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency": end - start
    }

def query_llm(prompt, temperature=0):
    if LLM_PROVIDER.lower() == "ollama":
        return query_ollama(prompt, temperature=temperature)
    elif LLM_PROVIDER.lower() == "openai":
        return query_openai(prompt, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")