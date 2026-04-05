from evaluation.runner import run_experiment
from evaluation.logger import save_results_json, save_results_csv
from evaluation.metrics import summarize_results
from methods.baseline import baseline_method
from methods.prompt_constraints import prompt_constraints_method
from methods.self_correction import self_correction_method
from methods.self_consistency import self_consistency_method
from methods.rag import rag_method


METHODS = {
    "1": ("baseline", baseline_method),
    "2": ("prompt_constraints", prompt_constraints_method),
    "3": ("self_correction", self_correction_method),
    "4": ("self_consistency", self_consistency_method),
    "5": ("rag", rag_method),
}


def main():
    print("\nSelect a method:")
    print("1. Baseline")
    print("2. Prompt Constraints")
    print("3. Self-Correction")
    print("4. Self-Consistency")
    print("5. RAG")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice not in METHODS:
        print("Invalid choice.")
        return

    method_name, method_fn = METHODS[choice]

    if method_fn is None:
        print(f"{method_name.upper()} method not yet implemented.")
        return

    max_q = input("Enter max questions (default 5): ").strip()
    max_questions = int(max_q) if max_q else 5

    results = run_experiment(
        method_fn=method_fn,
        method_name=method_name,
        max_questions=max_questions,
    )

    summary = summarize_results(results)

    json_path = save_results_json(results, f"{method_name}_results.json")
    csv_path = save_results_csv(results, f"{method_name}_results.csv")

    print(f"\nSaved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nSample result:")
    print(results[0])


if __name__ == "__main__":
    main()
