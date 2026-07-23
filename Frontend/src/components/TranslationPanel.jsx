import React, { useState } from "react";
import { translate } from "../api/client";

const SwapIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M7 16V4m0 0L3 8m4-4l4 4"/><path d="M17 8v12m0 0l4-4m-4 4l-4-4"/>
  </svg>
);

const ErrorIcon = () => (
  <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

export default function TranslationPanel({ model }) {
  const initialSrc = model.id === "translate_en_ar" ? "en" : "ar";
  const initialTgt = model.id === "translate_en_ar" ? "ar" : "en";

  const [srcLang, setSrcLang] = useState(initialSrc);
  const [tgtLang, setTgtLang] = useState(initialTgt);
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const LANG_NAME = { en: "English", ar: "العربية" };

  const handleSwap = () => {
    setSrcLang(tgtLang);
    setTgtLang(srcLang);
    setInputText(outputText);
    setOutputText("");
  };

  const handleTranslate = async () => {
    const trimmed = inputText.trim();
    if (!trimmed || loading) return;
    setError(null);
    setLoading(true);
    try {
      const res = await translate(trimmed, srcLang, tgtLang);
      setOutputText(res.data.translated_text);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Network Error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleTranslate();
  };

  return (
    <div>
      <div className="module-header anim-fade-up delay-0">
        <p className="module-eyebrow">Neural Machine Translation</p>
        <h2 className="module-title">{model.name}</h2>
        <p className="module-desc">
          MarianMT-powered translation between English and Arabic. High-fidelity output preserving semantic structure.
        </p>
      </div>

      <div className="trans-module">
        {/* Lang bar */}
        <div className="trans-lang-bar glass-sm anim-fade-up delay-1">
          <span className="lang-chip active">{LANG_NAME[srcLang]}</span>

          <button className="btn-swap" onClick={handleSwap} aria-label="Swap languages">
            <SwapIcon />
          </button>

          <span className="lang-chip active" style={{
            background: "var(--c-indigo-dim)",
            borderColor: "rgba(99,102,241,0.3)",
            color: "var(--c-violet)"
          }}>
            {LANG_NAME[tgtLang]}
          </span>

          <div style={{ flex: 1 }} />
          <span style={{ fontSize: "0.78rem", color: "var(--c-text-3)" }}>Ctrl+Enter to translate</span>
        </div>

        {/* Grid */}
        <div className="trans-grid anim-fade-up delay-2">
          <div className="glass-sm" style={{ overflow: "hidden" }}>
            <div style={{
              padding: "8px 16px",
              borderBottom: "1px solid var(--c-border)",
              fontSize: "0.75rem", fontWeight: 600,
              textTransform: "uppercase", letterSpacing: "0.07em",
              color: "var(--c-text-3)", background: "var(--c-bg-2)",
            }}>
              {LANG_NAME[srcLang]} — Source
            </div>
            <textarea
              className="input-pro"
              style={{
                border: "none", borderRadius: 0, background: "transparent",
                minHeight: "240px", direction: srcLang === "ar" ? "rtl" : "ltr",
                boxShadow: "none",
              }}
              placeholder={`Type in ${LANG_NAME[srcLang]}...`}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <div className="glass-sm" style={{ overflow: "hidden" }}>
            <div style={{
              padding: "8px 16px",
              borderBottom: "1px solid var(--c-border)",
              fontSize: "0.75rem", fontWeight: 600,
              textTransform: "uppercase", letterSpacing: "0.07em",
              color: "var(--c-text-3)", background: "var(--c-bg-2)",
              display: "flex", alignItems: "center", gap: "8px",
            }}>
              {LANG_NAME[tgtLang]} — Translation
              {loading && <span className="spinner" style={{ width: "14px", height: "14px", borderWidth: "1.5px" }} />}
            </div>
            <div
              className={`trans-output-box ${outputText ? "filled" : ""}`}
              style={{
                borderRadius: 0, border: "none",
                direction: tgtLang === "ar" ? "rtl" : "ltr",
                minHeight: "240px",
              }}
            >
              {outputText || (loading ? "Translating..." : "Translation will appear here...")}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="anim-fade-up delay-3" style={{ display: "flex", alignItems: "center", gap: "var(--sp-md)" }}>
          <button className="btn-primary" onClick={handleTranslate} disabled={!inputText.trim() || loading}>
            {loading ? <span className="spinner" /> : "Translate"}
          </button>
          {inputText && (
            <button className="btn-ghost" onClick={() => { setInputText(""); setOutputText(""); setError(null); }}>
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
    </div>
  );
}
