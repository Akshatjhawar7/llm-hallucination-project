from evaluation.runner import run_experiment
from evaluation.logger import save_results_json, save_results_csv
from evaluation.metrics import summarize_results
from methods.prompt_constraints import prompt_constraints_method

if __name__ == "__main__":
    results = run_experiment(
        method_fn = prompt_constraints_method, 
        method_name = "prompt_constraints_v1", 
        max_questions = 5
    )

    summary = summarize_results(results)

    json_path = save_results_json(results, "prompt_constraints_test.json")
    csv_path = save_results_csv(results, "prompt_constraints_test.csv")

    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSample result:")
    print(results[0])