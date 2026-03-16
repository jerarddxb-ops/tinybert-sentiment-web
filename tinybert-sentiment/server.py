import os
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import BertTokenizerFast, AutoModelForSequenceClassification

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_files")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
INDEX_FILE = os.path.join(FRONTEND_DIR, "index.html")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

class SentimentRequest(BaseModel):
    text: str

@app.get("/")
def serve_frontend():
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health():
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
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    sentiment_map = {
        0: "Negative 😞",
        1: "Neutral 😐",
        2: "Positive 🙂"
    }

    sentiment = sentiment_map.get(predicted_class, "Unknown")

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 1)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
