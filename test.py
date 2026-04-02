from data.loader import load_truthfulqa

df = load_truthfulqa(max_questions=5)

print("Shape:", df.shape)
print(df.head())
print("\nFirst question:")
print(df.iloc[0]["question"])
print("\nBest answer:")
print(df.iloc[0]["best_answer"])
