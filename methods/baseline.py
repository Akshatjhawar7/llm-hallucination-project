from utils.llm import query_llm


def clean_response(text):
    if not text:
        return ""
    return " ".join(text.strip().split())


def baseline_method(question):
    prompt = f"""Answer the following question.

Question: {question}

Answer:
"""
    output = query_llm(prompt)
    output["response"] = clean_response(output.get("response", ""))
    return output
