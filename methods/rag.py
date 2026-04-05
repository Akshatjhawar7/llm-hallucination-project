import time
import wikipediaapi
from utils.llm import query_llm

MAX_CONTEXT_CHARS = 1500

wiki = wikipediaapi.Wikipedia("HallucinationProject/1.0", "en")


def clean_response(text):
    if not text:
        return ""
    return " ".join(text.strip().split())


def retrieve_context(question):
    start = time.time()

    try:
        # Extract key terms by using the question directly as a page search
        # Try the full question first, then fall back to shorter terms
        words = question.strip().rstrip("?").split()

        # Try progressively shorter search terms
        attempts = [
            " ".join(words),           # full question
            " ".join(words[-4:]),       # last 4 words (often the subject)
            " ".join(words[-2:]),       # last 2 words
        ]

        for attempt in attempts:
            page = wiki.page(attempt)
            if page.exists():
                context = page.summary[:MAX_CONTEXT_CHARS]
                return {
                    "context": context,
                    "source": page.title,
                    "retrieval_latency": time.time() - start,
                    "success": True,
                }

        return {
            "context": "",
            "source": None,
            "retrieval_latency": time.time() - start,
            "success": False,
        }

    except Exception:
        return {
            "context": "",
            "source": None,
            "retrieval_latency": time.time() - start,
            "success": False,
        }


def build_rag_prompt(question, context):
    if not context:
        return f"""Answer the following question.
If you are not confident in the answer, reply exactly: I don't know.
Keep the answer to 1-2 sentences maximum.

Question: {question}

Answer:
"""

    return f"""You are answering a factual question using ONLY the provided evidence.

Evidence from Wikipedia:
\"\"\"
{context}
\"\"\"

Rules:
1. Answer ONLY using facts from the evidence above.
2. Do not add information from your own knowledge.
3. If the evidence does not contain enough information to answer, reply exactly: I don't know.
4. Keep the answer to 1-2 sentences maximum.
5. Do not guess or speculate.

Question: {question}

Answer:
"""


def rag_method(question):
    retrieval_info = retrieve_context(question)

    prompt = build_rag_prompt(question, retrieval_info["context"])
    llm_output = query_llm(prompt)

    final_answer = clean_response(llm_output.get("response", ""))
    total_latency = retrieval_info["retrieval_latency"] + (llm_output.get("latency", 0) or 0.0)

    return {
        "response": final_answer,
        "retrieved_context": retrieval_info["context"],
        "retrieval_source": retrieval_info["source"],
        "retrieval_latency": retrieval_info["retrieval_latency"],
        "retrieval_success": retrieval_info["success"],
        "prompt_tokens": llm_output.get("prompt_tokens"),
        "completion_tokens": llm_output.get("completion_tokens"),
        "total_tokens": llm_output.get("total_tokens"),
        "latency": total_latency,
    }
