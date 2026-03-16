from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load tokenizer and model from Hugging Face repo
tokenizer = AutoTokenizer.from_pretrained("JerardVenter/tinybert-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("JerardVenter/tinybert-sentiment")

# Pydantic model for incoming request
class SentimentRequest(BaseModel):
    text: str

# API endpoint
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return {"text": request.text, "predicted_class": predicted_class}

# Entry point for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets $PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port)
