import os
import sys
import joblib
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BACKEND_MODELS_DIR = os.path.join(os.path.dirname(__file__), "Models")

# Fallback for the Text_Summarization model that lives in the Chatbot Models dir
CHATBOT_MODELS_DIR = r"f:\M2.2\learning\Python_Machine\Chatbot\Models"

# Global dictionary to cache loaded models
_loaded_models: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_dir(candidate: str) -> str:
    """Return the candidate path if it exists, otherwise raise a helpful error."""
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Model directory not found: {candidate}")


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def get_model(model_name: str):
    """Load (and cache) the requested model. model_name must match one of the
    keys handled below."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    # -----------------------------------------------------------------------
    # Text Summarization  (flan-t5, trained model)
    # -----------------------------------------------------------------------
    if model_name == "Text_Summarization":
        manifest_path = os.path.join(
            CHATBOT_MODELS_DIR,
            "Text_Summarization", "results", "flan_t5_summarizer", "model_manifest.joblib",
        )
        # Also check backend-local path
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(
                BACKEND_MODELS_DIR,
                "flan_t5_summarizer", "model_manifest.joblib",
            )
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Text_Summarization manifest not found.")
        manifest = joblib.load(manifest_path)
        # The manifest["model_dir"] contains a broken relative path. 
        # The model files are actually in the exact same directory as the manifest.
        model_dir = os.path.dirname(manifest_path)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        _loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}

    # -----------------------------------------------------------------------
    # Marian MT  EN → AR
    # -----------------------------------------------------------------------
    elif model_name == "MarianMT_en_ar":
        model_dir = os.path.join(BACKEND_MODELS_DIR, "marian_mt_en_ar")
        if not os.path.exists(model_dir):
            # Fall back to HuggingFace hub name
            model_dir = "Helsinki-NLP/opus-mt-en-ar"
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = MarianMTModel.from_pretrained(model_dir)
        _loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}

    # -----------------------------------------------------------------------
    # Marian MT  AR → EN
    # -----------------------------------------------------------------------
    elif model_name == "MarianMT_ar_en":
        model_dir = os.path.join(BACKEND_MODELS_DIR, "marian_mt_ar_en")
        if not os.path.exists(model_dir):
            model_dir = "Helsinki-NLP/opus-mt-ar-en"
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = MarianMTModel.from_pretrained(model_dir)
        _loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}

    # -----------------------------------------------------------------------
    # Emotion Detection  (scikit-learn joblib artifact)
    # -----------------------------------------------------------------------
    elif model_name == "Emotion":
        artifact_path = os.path.join(BACKEND_MODELS_DIR, "emotion_model_artifacts.joblib")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Emotion model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        _loaded_models[model_name] = artifact  # dict with 'model', 'vectorizer', etc.

    # -----------------------------------------------------------------------
    # Sentiment Analysis  (scikit-learn joblib artifact)
    # -----------------------------------------------------------------------
    elif model_name == "Sentiment":
        artifact_path = os.path.join(BACKEND_MODELS_DIR, "sentiment_model_artifacts.joblib")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Sentiment model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        _loaded_models[model_name] = artifact

    # -----------------------------------------------------------------------
    # DistilGPT-2  (text generation)
    # -----------------------------------------------------------------------
    elif model_name == "DistilGPT2":
        model_dir = "distilgpt2"  # always load from HuggingFace hub
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        _loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}

    # -----------------------------------------------------------------------
    # Gemini  (API-based — no local weights to load)
    # -----------------------------------------------------------------------
    elif model_name == "Gemini":
        # We only configure the SDK here; the chat session is created per-request
        from dotenv import load_dotenv
        import google.generativeai as genai

        env_path = os.path.join(BACKEND_MODELS_DIR, "Gemini", ".env")
        load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment / .env file.")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        _loaded_models[model_name] = {"genai_model": gemini_model}

    else:
        raise NotImplementedError(f"Model loader for '{model_name}' is not implemented.")

    return _loaded_models[model_name]
