from transformers import pipeline
import os

# Path to your trained TinyBERT model
model_path = os.path.abspath("./tinybert-sentiment")

# Load pipeline with local model
sentiment_analyzer = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    return_all_scores=False,
    local_files_only=True  # IMPORTANT: only use local files
)

print("📝 Type a sentence and hit Enter. Type 'quit' to exit.")

while True:
    text = input(">> ")
    if text.lower() in ["quit", "exit"]:
        break
    result = sentiment_analyzer(text)
    print(f"Sentiment: {result[0]['label']} (score: {result[0]['score']:.2f})")