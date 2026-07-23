import os
import torch
import joblib
from transformers import MarianTokenizer, MarianMTModel

# =====================================================
# Paths
# =====================================================

MODEL_DIR = os.path.join("results", "marian_mt_en_ar")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "model_manifest.joblib")

# =====================================================
# Load Model Information
# =====================================================

if not os.path.exists(ARTIFACT_PATH):
    raise FileNotFoundError("model_manifest.joblib was not found.")

manifest = joblib.load(ARTIFACT_PATH)
max_length = manifest["max_length"]

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# Load Tokenizer and Model
# =====================================================

print("Loading model...")

tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
model = MarianMTModel.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

print("Model loaded successfully!")

# =====================================================
# Translation Function
# =====================================================

def translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )

    translation = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True
    )

    return translation

# =====================================================
# Interactive Testing
# =====================================================

print("\nEnglish → Arabic Translator")
print("Type 'exit' to quit.\n")

while True:
    sentence = input("English: ").strip()

    if sentence.lower() == "exit":
        break

    arabic = translate(sentence)

    print("Arabic :", arabic)
    print("-" * 60)