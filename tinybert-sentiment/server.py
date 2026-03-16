import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI
app = FastAPI()

# Get the directory where server.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to model folder
MODEL_PATH = os.path.join(BASE_DIR, "model_files")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Request schema
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

# Run locally or on Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
