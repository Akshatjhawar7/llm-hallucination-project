from utils.llm import query_llm
from evaluation.metrics import token_f1, is_idk_response

N_SAMPLES = 5
TEMPERATURE = 0.8
CONSENSUS_THRESHOLD = 0.4
IDK_DOMINANCE_THRESHOLD = 0.5


def clean_response(text):
    if not text:
        return ""
    return " ".join(text.strip().split())


def build_prompt(question):
    return f"""Answer the question in one sentence only.

Rules:
1. Use at most 15 words.
2. Be direct and factual.
3. Do not guess. If unsure, reply exactly: I don't know.
4. Do not add background or explanation.

Question: {question}

Answer:
"""


def sample_responses(question, n_samples, temperature):
    samples = []
    total_latency = 0.0
    prompt = build_prompt(question)

    for _ in range(n_samples):
        output = query_llm(prompt, temperature=temperature)
        response = clean_response(output.get("response", ""))
        latency = output.get("latency", 0) or 0.0
        samples.append(response)
        total_latency += latency

    return samples, total_latency


def compute_agreement_scores(responses):
    n = len(responses)
    if n <= 1:
        return [1.0] * n

    scores = []
    for i in range(n):
        pairwise = []
        for j in range(n):
            if i != j:
                pairwise.append(token_f1(responses[i], responses[j]))
        scores.append(sum(pairwise) / len(pairwise))

    return scores


def select_consensus_answer(responses, agreement_scores, threshold):
    if not responses:
        return "I don't know", 0.0, False

    # Check IDK dominance
    idk_count = sum(1 for r in responses if is_idk_response(r))
    if idk_count / len(responses) >= IDK_DOMINANCE_THRESHOLD:
        return "I don't know", 0.0, False

    # Filter out IDK responses from consensus voting
    non_idk = [(r, s) for r, s in zip(responses, agreement_scores) if not is_idk_response(r)]

    if not non_idk:
        return "I don't know", 0.0, False

    # Pick the response with highest agreement
    best_response, best_score = max(non_idk, key=lambda x: x[1])

    if best_score < threshold:
        return "I don't know", best_score, False

    return best_response, best_score, True


def self_consistency_method(question, n_samples=N_SAMPLES):
    samples, total_latency = sample_responses(question, n_samples, TEMPERATURE)
    agreement_scores = compute_agreement_scores(samples)
    final_answer, best_score, consensus_reached = select_consensus_answer(
        samples, agreement_scores, CONSENSUS_THRESHOLD
    )

    return {
        "response": final_answer,
        "all_samples": samples,
        "agreement_scores": agreement_scores,
        "best_agreement_score": best_score,
        "consensus_reached": consensus_reached,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency": total_latency,
    }
