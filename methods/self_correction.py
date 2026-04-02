from utils.llm import query_llm

def clean_response(text):
    if not text:
        return ""

    return " ".join(text.strip().split())

def generate_initial_response(question):
    prompt = f"""
Answer the following question briefly.

Rules:
1. Give a short, direct answer.
2. Do not add unnecessary information.
3. Answer in 1 or 2 sentences.

Question: {question}

Answer:
"""
    return query_llm(prompt)

def critique_answer(question, initial_answer):
    prompt = f"""
Check the following answer for factual risk.

Question: {question}

Initial Answer: {initial_answer}

Be strict.

Return exactly in this format:
Safe: yes or no
Risk: <short explanation>
Recommendation: <either a safer corrected answer, or exactly 'I don't know'>

Only mark Safe: yes if the answer is factually reliable and does not include unsupported claims.
"""
    return query_llm(prompt)

def revise_answer(question, initial_answer, critique):
    prompt = f"""
Produce the safest possible final answer to the question.

Question: {question}

Initial Answer: {initial_answer}

Critique: {critique}

Rules:
1. Remove any claim that is unsupported, speculative, misleading, or based on common misconception.
2. Keep only claims that are very likely to be true.
3. If there is any meaningful doubt, reply exactly: I don't know.
4. Do not preserve wording from the initial answer unless it is clearly safe.
5. Use at most 1 sentence.
6. Do not explain your reasoning.

Final Answer:
"""
    return query_llm(prompt)

def self_correction_method(question):
    initial_output = generate_initial_response(question)
    initial_answer = clean_response(initial_output.get("response", ""))

    critique_output = critique_answer(question, initial_answer)
    critique = clean_response(critique_output.get("response", ""))

    final_output = revise_answer(question, initial_answer, critique)
    final_answer = clean_response(final_output.get("response", ""))

    total_latency = (
        (initial_output.get("latency", 0) or 0.0) +
        (critique_output.get("latency", 0) or 0.0) +
        (final_output.get("latency", 0) or 0.0)
    )

    return {
        "response": final_answer,
        "initial_answer": initial_answer,
        "critique": critique,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency": total_latency,
    }