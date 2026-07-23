from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import get_model
import inference_utils

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
# Enable CORS for the React dev servers
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Unified ML Backend API v2 (Flask)"})

# ---------------------------------------------------------------------------
# Text Summarization
# ---------------------------------------------------------------------------
@app.route("/summarize", methods=["POST"])
def summarize_text():
    try:
        data = request.json
        text = data.get("text")
        max_length = data.get("max_length", 130)
        min_length = data.get("min_length", 30)

        if not text:
            return jsonify({"detail": "text is required"}), 400

        model_data = get_model("Text_Summarization")
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# ---------------------------------------------------------------------------
# Translation  (MarianMT  EN ↔ AR)
# ---------------------------------------------------------------------------
@app.route("/translate", methods=["POST"])
def translate_text():
    try:
        data = request.json
        text = data.get("text")
        source_lang = data.get("source_lang", "en")
        target_lang = data.get("target_lang", "ar")

        if not text:
            return jsonify({"detail": "text is required"}), 400

        if source_lang == "en" and target_lang == "ar":
            model_key = "MarianMT_en_ar"
        elif source_lang == "ar" and target_lang == "en":
            model_key = "MarianMT_ar_en"
        else:
            return jsonify({"detail": f"Unsupported language pair: {source_lang} -> {target_lang}"}), 400

        model_data = get_model(model_key)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        return jsonify({"translated_text": result, "source_lang": source_lang, "target_lang": target_lang})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# ---------------------------------------------------------------------------
# Emotion Detection
# ---------------------------------------------------------------------------
@app.route("/emotion", methods=["POST"])
def detect_emotion():
    try:
        data = request.json
        text = data.get("text")
        if not text:
            return jsonify({"detail": "text is required"}), 400

        artifact = get_model("Emotion")
        label = inference_utils.run_emotion_inference(artifact, text)

        return jsonify({"emotion": label})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------
@app.route("/sentiment", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.json
        text = data.get("text")
        if not text:
            return jsonify({"detail": "text is required"}), 400

        artifact = get_model("Sentiment")
        label = inference_utils.run_sentiment_inference(artifact, text)

        return jsonify({"sentiment": label})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.json
        prompt = data.get("prompt")
        max_length = data.get("max_length", 200)

        if not prompt:
            return jsonify({"detail": "prompt is required"}), 400

        model_data = get_model("DistilGPT2")
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# ---------------------------------------------------------------------------
# Gemini Chat
# ---------------------------------------------------------------------------
@app.route("/chat/gemini", methods=["POST"])
def gemini_chat():
    try:
        data = request.json
        prompt = data.get("prompt")
        history = data.get("history", [])

        if not prompt:
            return jsonify({"detail": "prompt is required"}), 400

        model_data = get_model("Gemini")
        gemini_model = model_data["genai_model"]

        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(prompt)
        reply = response.text

        updated_history = list(history) + [
            {"role": "user", "parts": [prompt]},
            {"role": "model", "parts": [reply]},
        ]

        return jsonify({"reply": reply, "history": updated_history})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
