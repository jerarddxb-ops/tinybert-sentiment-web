import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Since Render root directory is set to "tinybert-sentiment",
# the model folder is directly inside the working directory.
MODEL_PATH = "Model_files"

# Load tokenizer and model from local files
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Request body schema
class SentimentRequest(BaseModel):
    text: str

# Optional health check route
@app.get("/")
def root():
    return {"status": "ok", "message": "TinyBERT sentiment API is running"}

# Prediction endpoint
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

    return {
        "text": request.text,
        "predicted_class": predicted_class
    }

# Local run / Render compatibility
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
