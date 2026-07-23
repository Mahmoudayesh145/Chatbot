import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

// Text Summarization
export const summarize = (text, maxLength = 130, minLength = 30) =>
  api.post("/summarize", { text, max_length: maxLength, min_length: minLength });

// Translation  (source_lang + target_lang: "en" | "ar")
export const translate = (text, source_lang = "en", target_lang = "ar") =>
  api.post("/translate", { text, source_lang, target_lang });

// Emotion Detection
export const detectEmotion = (text) => api.post("/emotion", { text });

// Sentiment Analysis
export const analyzeSentiment = (text) => api.post("/sentiment", { text });

// Text Generation (DistilGPT-2)
export const generateText = (prompt, maxLength = 200) =>
  api.post("/generate", { prompt, max_length: maxLength });

// Gemini Chat  (history: [{role, parts: [string]}])
export const geminiChat = (prompt, history = []) =>
  api.post("/chat/gemini", { prompt, history });

export default api;
