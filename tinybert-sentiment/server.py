import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI
app = FastAPI()

# Path to model files inside your repo
MODEL_PATH = "tinybert-sentiment/Model_files"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Request format
class SentimentRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    inputs = tokenizer(request.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return {
        "text": request.text,
        "predicted_class": predicted_class
    }

# Run server (for local use and Render compatibility)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
