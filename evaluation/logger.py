import os
import json
import pandas as pd
from datetime import datetime

def ensure_results_dir():
    os.makedirs("results", exist_ok=True)

def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results_json(results, filename=None):
    ensure_results_dir()

    if filename is None:
        filename = f"results_{timestamp_str()}.json"

    path = os.path.join("results", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path

def save_results_csv(results, filename=None):
    ensure_results_dir()

    if filename is None:
        filename = f"results_{timestamp_str()}.csv"

    path = os.path.join("results", filename)
    
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, encoding="utf-8")

    return path