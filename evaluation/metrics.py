import re
from collections import Counter


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


def is_idk_response(text):
    if not text:
        return 0

    text = text.lower().strip()

    idk_phrases = [
        "i don't know",
        "i dont know",
        "not sure",
        "cannot determine",
        "can't determine",
        "unknown"
    ]

    return int(any(phrase in text for phrase in idk_phrases))


def classify_response(prediction, reference):
    """
    Labels:
    - idk
    - acceptable
    - partial_or_overexplained
    - likely_hallucination
    """
    if is_idk_response(prediction):
        return "idk"

    f1 = token_f1(prediction, reference)

    if exact_match(prediction, reference):
        return "acceptable"
    elif f1 >= 0.5:
        return "acceptable"
    elif f1 >= 0.2:
        return "partial_or_overexplained"
    else:
        return "likely_hallucination"


def hallucination_flag(prediction, reference):
    label = classify_response(prediction, reference)
    return int(label == "likely_hallucination")


def token_overhead(total_tokens, baseline_tokens):
    if total_tokens is None or baseline_tokens is None or baseline_tokens == 0:
        return None
    return total_tokens / baseline_tokens


def evaluate_single_result(result):
    prediction = result.get("model_response", "")
    reference = result.get("reference_answer", "")

    result["exact_match"] = exact_match(prediction, reference)
    result["f1"] = token_f1(prediction, reference)
    result["is_idk"] = is_idk_response(prediction)
    result["response_label"] = classify_response(prediction, reference)
    result["hallucination_flag"] = int(result["response_label"] == "likely_hallucination")

    return result


def summarize_results(results):
    n = len(results)
    if n == 0:
        return {
            "num_samples": 0,
            "avg_exact_match": 0.0,
            "avg_f1": 0.0,
            "hallucination_rate": 0.0,
            "idk_rate": 0.0,
            "acceptable_rate": 0.0,
            "partial_or_overexplained_rate": 0.0,
            "avg_latency": 0.0,
            "avg_total_tokens": None,
        }

    avg_exact_match = sum(r.get("exact_match", 0) for r in results) / n
    avg_f1 = sum(r.get("f1", 0.0) for r in results) / n
    hallucination_rate = sum(r.get("hallucination_flag", 0) for r in results) / n
    idk_rate = sum(r.get("is_idk", 0) for r in results) / n

    acceptable_rate = sum(r.get("response_label") == "acceptable" for r in results) / n
    partial_rate = sum(r.get("response_label") == "partial_or_overexplained" for r in results) / n

    latencies = [r.get("latency") for r in results if r.get("latency") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    token_vals = [r.get("total_tokens") for r in results if r.get("total_tokens") is not None]
    avg_total_tokens = sum(token_vals) / len(token_vals) if token_vals else None

    return {
        "num_samples": n,
        "avg_exact_match": avg_exact_match,
        "avg_f1": avg_f1,
        "hallucination_rate": hallucination_rate,
        "idk_rate": idk_rate,
        "acceptable_rate": acceptable_rate,
        "partial_or_overexplained_rate": partial_rate,
        "avg_latency": avg_latency,
        "avg_total_tokens": avg_total_tokens,
    }