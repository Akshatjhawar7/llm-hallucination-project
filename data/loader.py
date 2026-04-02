from datasets import load_dataset
import pandas as pd

def load_truthfulqa(split="validation", max_questions=None):
    """
    Loads TruthfulQA dataset and returns a DataFrame.
    Args:
        split: dataset split to use
        max_questions: optional cap for quick testing
    """
    dataset = load_dataset("truthful_qa", "generation")
    ds = dataset[split]
    df = pd.DataFrame(ds)

    keep_cols = [
        "question",
        "best_answer",
        "correct_answers",
        "incorrect_answers",
        "category",
    ]

    df = df[keep_cols]

    if max_questions is not None:
        df = df.head(max_questions).copy()

    return df

def load_truthfulqa_as_records(max_questions=None):
    df = load_truthfulqa(max_questions=max_questions)
    return df.to_dict(orient="records")