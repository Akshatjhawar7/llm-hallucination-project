from tqdm import tqdm
from data.loader import load_truthfulqa_as_records
from evaluation.metrics import evaluate_single_result, summarize_results

def run_experiment(method_fn, method_name, max_questions=10):
    dataset = load_truthfulqa_as_records(max_questions=max_questions)
    results = []

    for i, item in enumerate(tqdm(dataset, desc=f"Running {method_name}")):
        question = item["question"]
        reference_answer = item["best_answer"]
        category = item["category"]

        try:
            output = method_fn(question)

            result = {
                "id": i,
                "method": method_name,
                "question": question,
                "reference_answer": reference_answer,
                "model_response": output.get("response", ""),
                "prompt_tokens": output.get("prompt_tokens"),
                "completion_tokens": output.get("completion_tokens"),
                "total_tokens": output.get("total_tokens"),
                "latency": output.get("latency"),
                "category": category,
                "error": None,
            }
        except Exception as e:
            result = {
                "id": i,
                "method": method_name,
                "question": question,
                "reference_answer": reference_answer,
                "model_response": "",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency": None,
                "category": category,
                "error": str(e),
            }

        result = evaluate_single_result(result)
        results.append(result)

    return results