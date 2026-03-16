# predict.py
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

# 1️⃣ Load model and tokenizer
model_path = "./tinybert-sentiment"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Put model in evaluation mode
model.eval()

# 2️⃣ Mapping IDs back to labels
id2label = model.config.id2label

# 3️⃣ Terminal loop for user input
print("Enter a sentence to predict sentiment ('quit' to exit):")
while True:
    sentence = input(">>> ")
    if sentence.lower() in ["quit", "exit"]:
        break
    
    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
    # Get logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted class
    predicted_id = int(torch.argmax(logits, dim=-1))
    predicted_label = id2label[predicted_id]
    
    # Print result neatly
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {predicted_label}\n")