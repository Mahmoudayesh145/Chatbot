import torch
import os
import re
import pandas as pd
import joblib
from datasets import load_dataset
from ftfy import fix_text
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import nltk

import numpy as np

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

DATA_DIR = "data"
TRAIN_RAW_PATH = os.path.join(DATA_DIR, "cnn_dailymail_dataset.csv")
VALIDATION_RAW_PATH = os.path.join(DATA_DIR, "cnn_dailymail_validation_dataset.csv")
TEST_RAW_PATH = os.path.join(DATA_DIR, "cnn_dailymail_test_dataset.csv")

TRAIN_CLEAN_PATH = os.path.join(DATA_DIR, "cnn_dailymail_dataset_preprocessed.csv")
VALIDATION_CLEAN_PATH = os.path.join(DATA_DIR, "cnn_dailymail_validation_dataset_preprocessed.csv")
TEST_CLEAN_PATH = os.path.join(DATA_DIR, "cnn_dailymail_test_dataset_preprocessed.csv")
MODEL_DIR = os.path.join("results1", "flan_t5_summarizer")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "model_manifest.joblib")
tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

rouge = evaluate.load("rouge")


if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

    # Limiting data to avoid massive RAM usage, but keeping enough for the model to learn (5000 examples)
    train_dataset = dataset["train"].select(range(8000)).to_pandas()
    validation_dataset = dataset["validation"].select(range(700)).to_pandas()
    test_dataset = dataset["test"].select(range(700)).to_pandas()

    train_dataset.to_csv(TRAIN_RAW_PATH, index=False)
    validation_dataset.to_csv(VALIDATION_RAW_PATH, index=False)
    test_dataset.to_csv(TEST_RAW_PATH, index=False)

def preprocess_data(df):
    df = df.dropna(subset=["article", "highlights"]).copy()
    df = df.drop_duplicates(subset=["article", "highlights"])
    df = df[df["article"].str.strip() != ""]
    df = df[df["highlights"].str.strip() != ""]
    df["article"] = df["article"].str.replace(
        r"<.*?>",
        "",
        regex=True
    )
    df["highlights"] = df["highlights"].str.replace(
        r"<.*?>",
        "",
        regex=True
    )
    df["article"] = (
        df["article"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )
    df["highlights"] = (
        df["highlights"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )
    df["article"] = df["article"].apply(fix_text)
    df["highlights"] = df["highlights"].apply(fix_text)
    return df


def load_or_preprocess_csv(raw_path, clean_path):
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)

    df = pd.read_csv(raw_path)
    df = preprocess_data(df)
    df.to_csv(clean_path, index=False)
    return df

max_input_length = 512
max_target_length = 256

def tokenize_function(examples):

    inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True
    )

    inputs["labels"] = labels["input_ids"]

    return inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # In some models, predictions might also contain -100 if padded
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    predictions = np.where(
        predictions != -100,
        predictions,
        tokenizer.pad_token_id
    )

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    # Replace ignored tokens (-100) with pad token
    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id
    )

    # Decode labels
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Convert to percentages
    result = {k: round(v * 100, 2) for k, v in result.items()}

    return result

train_df = load_or_preprocess_csv(TRAIN_RAW_PATH, TRAIN_CLEAN_PATH)

validation_df = load_or_preprocess_csv(VALIDATION_RAW_PATH, VALIDATION_CLEAN_PATH)

test_df = load_or_preprocess_csv(TEST_RAW_PATH, TEST_CLEAN_PATH)


def add_prompt_columns(examples):
    prompt = (
        "Write a comprehensive summary of the following article. "
        "Ensure you include the main event, public reaction, any criticisms, and future plans.\n\nArticle:\n"
    )
    return {
        "input_text": [prompt + article for article in examples["article"]],
        "target_text": examples["highlights"],
    }


train_dataset = load_dataset("csv", data_files=TRAIN_CLEAN_PATH, split="train")
validation_dataset = load_dataset("csv", data_files=VALIDATION_CLEAN_PATH, split="train")
test_dataset = load_dataset("csv", data_files=TEST_CLEAN_PATH, split="train")

train_dataset = train_dataset.map(add_prompt_columns, batched=True)
validation_dataset = validation_dataset.map(add_prompt_columns, batched=True)
test_dataset = test_dataset.map(add_prompt_columns, batched=True)

train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)
validation_tokenized = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=validation_dataset.column_names,
)
test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=test_dataset.column_names,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)


model.generation_config.no_repeat_ngram_size = 3
model.generation_config.repetition_penalty = 1.2

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1, 
    weight_decay=0.01,
    warmup_steps=100,

    logging_strategy="steps",
    logging_steps=100,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=max_target_length,
    generation_num_beams=8,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,

    args=training_args,

    train_dataset=train_tokenized,

    eval_dataset=validation_tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)

trainer.train()

validation_results = trainer.evaluate(validation_tokenized)
test_results = trainer.evaluate(test_tokenized)

os.makedirs(MODEL_DIR, exist_ok=True)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
joblib.dump(
    {
        "model_dir": MODEL_DIR,
        "tokenizer_dir": MODEL_DIR,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
        "generation_max_length": max_target_length,
        "generation_num_beams": 8,
    },
    ARTIFACT_PATH,
)

print("Validation results:")
print(validation_results)
print("Test results:")
print(test_results)