import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizerFast, AutoModelForSequenceClassification

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_files")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

class SentimentRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "TinyBERT sentiment API is running"}

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    sentiment_map = {
        0: "Negative 😞",
        1: "Neutral 😐",
        2: "Positive 🙂"
    }

    sentiment = sentiment_map.get(predicted_class, "Unknown")

    return {
        "text": request.text,
        "sentiment": sentiment
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
