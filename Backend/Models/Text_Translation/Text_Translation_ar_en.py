import os
import torch
import joblib
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "translation_ar_en.csv")
MODEL_DIR = os.path.join("results3", "marian_mt_ar_en")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "model_manifest.joblib")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 1. Load dataset (Arabic to English only)
print("Loading dataset...")
dataset = load_dataset("Helsinki-NLP/opus-100", "ar-en")

train_data = dataset["train"].select(range(40000))

processed_data = {"source": [], "target": []}
for item in train_data:
    translation = item["translation"]
    if "en" in translation and "ar" in translation:
        processed_data["source"].append(translation["ar"])
        processed_data["target"].append(translation["en"])

df = pd.DataFrame(processed_data)
df.to_csv(CSV_PATH, index=False)
train_dataset = Dataset.from_pandas(df)

# 2. Load accurate Arabic-to-English base model
model_name = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

max_length = 128

def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

# 3. Setup fast training (1 epoch)
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1, # Reduced to 1 epoch for speed
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# 4. Save results
os.makedirs(MODEL_DIR, exist_ok=True)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
joblib.dump(
    {
        "model_dir": MODEL_DIR,
        "max_length": max_length,
    },
    ARTIFACT_PATH,
)
print(f"Model saved to {MODEL_DIR}")