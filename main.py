from utils.llm import query_llm
from utils.config import LLM_PROVIDER

if __name__ == "__main__":
    prompt = "Answer in 1 word only. What is the capital of France?"
    result = query_llm(prompt)

    print(f"Provider: {LLM_PROVIDER}")
    print(f"Response: {result['response']}")
    print(f"Prompt Tokens: {result['prompt_tokens']}")
    print(f"Completion Tokens: {result['completion_tokens']}")
    print(f"Total Tokens: {result['total_tokens']}")
    print(f"Latency: {round(result['latency'], 2)} seconds")