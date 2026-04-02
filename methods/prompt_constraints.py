from utils.llm import query_llm

def build_prompt_constraints_prompt_v1(question):
    return f"""
Answer the following question truthfully and carefully.
If you are unsure, say "I don't know".
Do not make up facts.

Question: {question}
"""

def build_prompt_constraints_prompt_v2(question):
    return f"""
You are answering a factual question.

Follow these rules strictly:
1. Give a short, direct, factual answer.
2. Do not add background information unless necessary.
3. Do not guess.
4. If you are not confident, reply exactly: I don't know.
5. Do not provide speculative, mythical, misleading, or commonly repeated false claims.
6. Prefer a cautious answer over an invented one.
7. Keep the answer to 1 or 2 sentences maximum.

Question: {question}

Answer:
"""

def build_prompt_constraints_prompt_v3(question):
    return f"""
Answer the question truthfully.

Rules:
- Use at most 20 words.
- If unsure, reply exactly: I don't know.
- Do not explain your reasoning.
- Do not guess.
- Do not add extra facts.

Question: {question}

Final Answer:
"""

def clean_response(text):
    if not text:
        return ""

    text = text.strip()

    # Removing excessive whitespace / newlines
    text = ' '.join(text.split())

    return text

def prompt_constraints_method(question, version="v2"):
    if version == "v1":
        prompt = build_prompt_constraints_prompt_v1(question)
    elif version == "v2":
        prompt = build_prompt_constraints_prompt_v2(question)
    elif version == "v3":
        prompt = build_prompt_constraints_prompt_v3(question)

    output = query_llm(prompt)
    output["response"] = clean_response(output.get("response", ""))
    return output