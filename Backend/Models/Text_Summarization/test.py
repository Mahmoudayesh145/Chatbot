import os

import joblib
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MANIFEST_PATH = os.path.join("results", "flan_t5_summarizer", "model_manifest.joblib")
DEFAULT_TEXT = (
    "Summarize:"
    "A large city park reopened on Saturday after being closed for nearly six months because of renovation work. Hundreds of local residents visited the park during its first day, enjoying the new walking paths, playgrounds, gardens, and picnic areas.City officials said the project cost $12 million and included planting more than 500 trees, improving lighting, repairing the lake, and building new sports courts. According to the mayor, the goal was to create a safer and more enjoyable place for families and visitors.Many residents praised the improvements. Parents said the larger playgrounds gave children more space to play, while runners appreciated the wider paths. Several local businesses also welcomed the reopening, saying they expected more customers because of the increased number of visitors.Not everyone was satisfied with the project. Some citizens argued that the renovation took too long and exceeded the original budget. Others believed that the city should have spent more money on public transportation and road repairs instead of the park.Despite the criticism, city officials said the project would benefit the community for many years. They announced that free outdoor concerts, sports events, and environmental education programs would be held in the park throughout the summer."
)


def load_artifacts():
    manifest = joblib.load(MANIFEST_PATH)
    model_dir = manifest["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model, tokenizer, manifest


def summarize_text(text, model, tokenizer, generation_max_length=128, generation_num_beams=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(
        f"Summarize:\n{text}",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=generation_max_length,
            num_beams=generation_num_beams,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer, manifest = load_artifacts()
    summary = summarize_text(
        DEFAULT_TEXT,
        model,
        tokenizer,
        generation_max_length=manifest["generation_max_length"],
        generation_num_beams=manifest["generation_num_beams"],
    )
    print("Input:")
    print(DEFAULT_TEXT)
    print()
    print("Summary:")
    print(summary)
