import React, { useState, useEffect } from "react";

export const MODELS = [
  {
    id: "gemini",
    name: "Gemini AI",
    desc: "Full conversational AI with multi-turn memory. Powered by Google Gemini 2.5 Flash.",
    tag: "Generative",
    gradient: "linear-gradient(90deg, #6366f1, #818cf8)",
    accent: "rgba(99,102,241,0.12)",
  },
  {
    id: "summarize",
    name: "Text Summarizer",
    desc: "Extracts the essential meaning from long-form text using a fine-tuned Flan-T5 model.",
    tag: "Seq2Seq",
    gradient: "linear-gradient(90deg, #22d3ee, #67e8f9)",
    accent: "rgba(34,211,238,0.10)",
  },
  {
    id: "translate_en_ar",
    name: "EN → AR Translation",
    desc: "High-fidelity English-to-Arabic neural machine translation via MarianMT.",
    tag: "Translation",
    gradient: "linear-gradient(90deg, #a78bfa, #c4b5fd)",
    accent: "rgba(167,139,250,0.10)",
  },
  {
    id: "translate_ar_en",
    name: "AR → EN Translation",
    desc: "High-fidelity Arabic-to-English neural machine translation via MarianMT.",
    tag: "Translation",
    gradient: "linear-gradient(90deg, #a78bfa, #c4b5fd)",
    accent: "rgba(167,139,250,0.10)",
  },
  {
    id: "emotion",
    name: "Emotion Detection",
    desc: "Classifies text into 15 fine-grained emotional categories with probability scores.",
    tag: "Classification",
    gradient: "linear-gradient(90deg, #fbbf24, #fde68a)",
    accent: "rgba(251,191,36,0.10)",
  },
  {
    id: "sentiment",
    name: "Sentiment Analysis",
    desc: "Determines positive, neutral, or negative polarity using TF-IDF + VADER features.",
    tag: "Classification",
    gradient: "linear-gradient(90deg, #34d399, #6ee7b7)",
    accent: "rgba(52,211,153,0.10)",
  },
  {
    id: "generate",
    name: "Text Generator",
    desc: "Open-ended autoregressive text generation using DistilGPT-2.",
    tag: "Generation",
    gradient: "linear-gradient(90deg, #f87171, #fca5a5)",
    accent: "rgba(248,113,113,0.10)",
  },
];

const ArrowIcon = () => (
  <svg className="model-card-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 12h14M12 5l7 7-7 7"/>
  </svg>
);

export default function ModelSelectModal({ onSelect }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 10); return () => clearTimeout(t); }, []);

  return (
    <div className="modal-backdrop">
      <div className="modal-panel">

        <div className="modal-header">
          <p className="modal-eyebrow">Select a Model</p>
          <h2 className="modal-title">What would you like to do?</h2>
          <p className="modal-sub">Select a machine learning model to begin your session. Each model is production-ready and inference-optimised.</p>
        </div>

        <div className="model-grid">
          {MODELS.map((m, i) => (
            <button
              key={m.id}
              className="model-card"
              style={{
                "--card-accent": m.accent,
                "--card-gradient": m.gradient,
                animationDelay: `${i * 0.06}s`,
              }}
              onClick={() => onSelect(m)}
            >
              <div className="model-card-accent-bar" />
              <div className="model-card-body">
                <div className="model-card-name">{m.name}</div>
                <div className="model-card-desc">{m.desc}</div>
                <div className="model-card-footer">
                  <span className="model-card-tag">{m.tag}</span>
                  <ArrowIcon />
                </div>
              </div>
            </button>
          ))}
        </div>

      </div>
    </div>
  );
}
