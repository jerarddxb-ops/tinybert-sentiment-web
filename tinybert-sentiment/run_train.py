import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# 1️⃣ Load dataset
dataset = load_dataset("json", data_files="./dataset.json")
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Explicitly define all labels
label_list = ["Negative", "Neutral", "Positive"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Map string labels to integers
def map_labels(example):
    example["label"] = label2id[example["label"]]
    return example

train_dataset = train_dataset.map(map_labels)
test_dataset = test_dataset.map(map_labels)

# 2️⃣ Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3️⃣ Tokenize datasets
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Remove original text column
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Set PyTorch format
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 4️⃣ Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 5️⃣ Define metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 6️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./tinybert-sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",       # must match save_strategy if load_best_model_at_end=True
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

# 7️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 8️⃣ Train
trainer.train()

# 9️⃣ Save model and tokenizer
trainer.save_model("./tinybert-sentiment")
tokenizer.save_pretrained("./tinybert-sentiment")

print("✅ Training complete. Model saved to ./tinybert-sentiment")