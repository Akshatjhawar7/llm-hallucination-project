from utils.llm import query_llm

def prompt_constraints_method(question):
    prompt = f"""
Answer the following question truthfully and carefully.
If you are unsure, say "I don't know".
Do not make up facts.
    
Question: {question}
"""
    return query_llm(prompt)