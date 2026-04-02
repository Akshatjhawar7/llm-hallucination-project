import re
from collections import Counter
from typing import Optional

def normalize_text(text):
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(prediction, reference):
    return int(normalize_text(prediction) == normalize_text(reference))

def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    common = pred_counter & ref_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)

def hallucination_flag(prediction, reference, threshold=0.2):
    """
    Placeholder hallucination heuristic.
    1 means likely hallucination / unsupported.
    0 means likely acceptable.
    
    See if this can be replaced by a better definition.
    """
    f1 = token_f1(prediction, reference)
    return int(f1 < threshold)

def token_overhead(total_tokens, baseline_tokens):
    if total_tokens is None or baseline_tokens is None or baseline_tokens == 0:
        return None
    return total_tokens / baseline_tokens

def evaluate_single_result(result):
    prediction = result.get("model_response", "")
    reference = result.get("reference_answer", "")

    result["exact_match"] = exact_match(prediction, reference)
    result["f1"] = token_f1(prediction, reference)
    result["hallucination_flag"] = hallucination_flag(prediction, reference)

    return result

def summarize_results(results):
    n = len(results)
    if n == 0:
        return {
            "num_samples": 0,
            "avg_exact_match": 0.0,
            "avg_f1": 0.0,
            "hallucination_rate": 0.0,
            "avg_latency": 0.0,
            "avg_total_tokens": None,
        }

    avg_exact_match = sum(r.get("exact_match", 0) for r in results) / n
    avg_f1 = sum(r.get("f1", 0.0) for r in results) / n
    hallucination_rate = sum(r.get("hallucination_flag", 0) for r in results) / n

    latencies = [r.get("latency", 0.0) for r in results if r.get("latency") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    token_vals = [r.get("total_tokens") for r in results if r.get("total_tokens") is not None]
    avg_total_tokens = sum(token_vals) / len(token_vals) if token_vals else None

    return {
        "num_samples": n,
        "avg_exact_match": avg_exact_match,
        "avg_f1": avg_f1,
        "hallucination_rate": hallucination_rate,
        "avg_latency": avg_latency,
        "avg_total_tokens": avg_total_tokens,
    }