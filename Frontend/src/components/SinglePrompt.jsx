import React, { useState } from "react";
import { summarize, detectEmotion, analyzeSentiment, generateText } from "../api/client";

const ErrorIcon = () => (
  <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

const MODEL_CONFIG = {
  summarize: {
    eyebrow: "Abstractive NLP",
    title: "Text Summarizer",
    desc: "Condenses long-form text into a precise, fluent summary using a fine-tuned Flan-T5 model.",
    placeholder: "Paste a long article, document, or paragraph you want summarised...",
    action: "Generate Summary",
    minRows: 8,
  },
  emotion: {
    eyebrow: "Emotion Classification",
    title: "Emotion Detection",
    desc: "Classifies text into 15 fine-grained emotional categories with probability scores.",
    placeholder: "Enter any text to detect the underlying emotional tone...",
    action: "Detect Emotion",
    minRows: 4,
  },
  sentiment: {
    eyebrow: "Sentiment Classification",
    title: "Sentiment Analysis",
    desc: "Determines the positive, neutral, or negative polarity of a text sample.",
    placeholder: "Enter a review, comment, or statement to analyse its sentiment...",
    action: "Analyse Sentiment",
    minRows: 4,
  },
  generate: {
    eyebrow: "Autoregressive Generation",
    title: "Text Generator",
    desc: "Continues your input with coherent generated text using DistilGPT-2.",
    placeholder: "Start a sentence or paragraph and let the model continue it...",
    action: "Generate Text",
    minRows: 4,
  },
};

export default function SinglePrompt({ model }) {
  const cfg = MODEL_CONFIG[model.id] || MODEL_CONFIG.generate;
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      let res;
      if (model.id === "summarize") {
        res = await summarize(trimmed);
        setResult({ type: "text", content: res.data.summary });
      } else if (model.id === "emotion") {
        res = await detectEmotion(trimmed);
        setResult({ type: "label", label: res.data.emotion, scores: res.data.scores });
      } else if (model.id === "sentiment") {
        res = await analyzeSentiment(trimmed);
        setResult({ type: "label", label: res.data.sentiment, scores: res.data.scores });
      } else if (model.id === "generate") {
        res = await generateText(trimmed);
        setResult({ type: "text", content: res.data.generated_text });
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Network Error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSubmit();
  };

  return (
    <div>
      {/* Module Header — staggered cascade */}
      <div className="module-header anim-fade-up delay-0">
        <p className="module-eyebrow">{cfg.eyebrow}</p>
        <h2 className="module-title">{cfg.title}</h2>
        <p className="module-desc">{cfg.desc}</p>
      </div>

      {/* Glass Panel — glass-hover for lift on hover */}
      <div className="glass glass-hover anim-fade-up delay-1" style={{ padding: "var(--sp-xl)" }}>

        <div className="field-group">
          <div className="field-label">
            <span className="field-label-text">Input</span>
            <span className="field-label-hint">{text.length} chars · Ctrl+Enter to run</span>
          </div>
          <textarea
            className="input-pro"
            style={{ minHeight: `${cfg.minRows * 24}px` }}
            placeholder={cfg.placeholder}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKey}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "var(--sp-md)" }}>
          <button className="btn-primary" onClick={handleSubmit} disabled={!text.trim() || loading}>
            {loading ? <span className="spinner" /> : cfg.action}
          </button>
          {text && (
            <button className="btn-ghost" onClick={() => { setText(""); setResult(null); setError(null); }}>
              Clear
            </button>
          )}
        </div>

        {error && (
          <div className="error-panel">
            <ErrorIcon />
            <span className="error-text">{error}</span>
          </div>
        )}
      </div>

      {result && (
        <div className="result-panel anim-fade-up delay-0">
          <div className="result-panel-header">
            <span className="result-panel-title">
              <span className="result-panel-dot" />
              Output
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--c-text-3)" }}>{model.name}</span>
          </div>
          <div className="result-panel-body">
            {result.type === "label" && (
              <div className="result-badge-row">
                <div className="result-badge">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                  {result.label}
                </div>
              </div>
            )}
            {result.type === "text" && (
              <p className="result-text">{result.content}</p>
            )}
            {result.scores && Object.keys(result.scores).length > 0 && (
              <div className="score-chips">
                {Object.entries(result.scores).map(([k, v]) => (
                  <span key={k} className="score-chip">{k}: {(v * 100).toFixed(1)}%</span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
