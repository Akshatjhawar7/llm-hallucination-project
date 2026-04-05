from evaluation.runner import run_experiment
from evaluation.logger import save_results_json, save_results_csv
from evaluation.metrics import summarize_results
from methods.baseline import baseline_method
from methods.prompt_constraints import prompt_constraints_method
from methods.self_correction import self_correction_method
from methods.self_consistency import self_consistency_method
from methods.rag import rag_method


METHODS = [
    ("baseline", baseline_method),
    ("prompt_constraints", prompt_constraints_method),
    ("self_correction", self_correction_method),
    ("self_consistency", self_consistency_method),
    ("rag", rag_method),
]


def run_all(max_questions=None):
    all_summaries = {}

    for method_name, method_fn in METHODS:
        print(f"\n{'='*60}")
        print(f"Running: {method_name}")
        print(f"{'='*60}")

        results = run_experiment(
            method_fn=method_fn,
            method_name=method_name,
            max_questions=max_questions,
        )

        summary = summarize_results(results)
        all_summaries[method_name] = summary

        json_path = save_results_json(results, f"{method_name}_results.json")
        csv_path = save_results_csv(results, f"{method_name}_results.csv")

        print(f"Saved: {json_path}, {csv_path}")
        print(f"Hallucination rate: {summary['hallucination_rate']:.2%}")
        print(f"IDK rate: {summary['idk_rate']:.2%}")
        print(f"F1: {summary['avg_f1']:.4f}")

    # Print final comparison table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    header = f"{'Metric':<30}"
    for name, _ in METHODS:
        header += f"{name:>18}"
    print(header)
    print("-" * (30 + 18 * len(METHODS)))

    metrics_to_show = [
        ("num_samples", "Samples"),
        ("hallucination_rate", "Hallucination Rate"),
        ("idk_rate", "IDK Rate"),
        ("acceptable_rate", "Acceptable Rate"),
        ("partial_or_overexplained_rate", "Partial Rate"),
        ("avg_exact_match", "Avg Exact Match"),
        ("avg_f1", "Avg F1"),
        ("avg_latency", "Avg Latency (s)"),
    ]

    for key, label in metrics_to_show:
        row = f"{label:<30}"
        for name, _ in METHODS:
            val = all_summaries[name].get(key)
            if val is None:
                row += f"{'N/A':>18}"
            elif isinstance(val, float):
                row += f"{val:>18.4f}"
            else:
                row += f"{val:>18}"
        print(row)


if __name__ == "__main__":
    run_all(max_questions=None)
