import React, { useState, useRef, useEffect } from "react";
import { geminiChat } from "../api/client";

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

const ClearIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
  </svg>
);

const SUGGESTIONS = [
  "Explain quantum entanglement in simple terms",
  "Write a haiku about machine learning",
  "What are the key differences between REST and GraphQL?",
  "Give me 5 productivity tips for developers",
];

export default function GeminiChat({ model }) {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const autoResize = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  };

  const sendMessage = async (text) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setError(null);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    setLoading(true);

    try {
      const res = await geminiChat(trimmed, history);
      const { reply, history: newHistory } = res.data;
      setMessages((prev) => [...prev, { role: "model", text: reply }]);
      setHistory(newHistory);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Network Error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setHistory([]);
    setError(null);
  };

  const isEmpty = messages.length === 0 && !loading;

  return (
    <div className="chat-module anim-fade-up delay-0">
      {/* Chat header */}
      <div className="chat-module-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div className="chat-module-title">Gemini AI — Conversational Session</div>
          <div className="chat-module-sub">Multi-turn memory · {messages.length} message{messages.length !== 1 ? "s" : ""} · Press Shift+Enter for new line</div>
        </div>
        {!isEmpty && (
          <button className="btn-ghost" onClick={clearChat} style={{ gap: "6px", fontSize: "0.8rem" }}>
            <ClearIcon /> Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {isEmpty ? (
          <div className="chat-empty">
            <div className="chat-empty-title">Start a conversation</div>
            <div className="chat-empty-sub">Ask anything — Gemini has full multi-turn memory.</div>

            <div style={{ marginTop: "var(--sp-xl)", display: "flex", flexDirection: "column", gap: "8px", maxWidth: "400px", margin: "var(--sp-xl) auto 0" }}>
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(s)}
                  style={{
                    textAlign: "left",
                    padding: "10px 16px",
                    background: "var(--c-bg-2)",
                    border: "1px solid var(--c-border)",
                    borderRadius: "var(--r-md)",
                    color: "var(--c-text-2)",
                    fontSize: "0.875rem",
                    cursor: "pointer",
                    transition:
                      "background var(--dur-fast) var(--ease-out), " +
                      "border-color var(--dur-fast) var(--ease-out), " +
                      "color var(--dur-fast) var(--ease-out), " +
                      "transform var(--dur-fast) var(--ease-out)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--c-border-med)";
                    e.currentTarget.style.color = "var(--c-text)";
                    e.currentTarget.style.background = "var(--c-bg-3)";
                    e.currentTarget.style.transform = "translateX(4px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--c-border)";
                    e.currentTarget.style.color = "var(--c-text-2)";
                    e.currentTarget.style.background = "var(--c-bg-2)";
                    e.currentTarget.style.transform = "translateX(0)";
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <div key={i} className={`msg ${msg.role}`}>
                <div className="msg-bubble">{msg.text}</div>
              </div>
            ))}

            {loading && (
              <div className="msg model">
                <div className="typing-indicator">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              </div>
            )}
          </>
        )}

        {error && (
          <div style={{ margin: "0 auto" }}>
            <div className="error-panel">
              <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span className="error-text">{error}</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Composer */}
      <div className="chat-composer">
        <div className="composer-inner">
          <textarea
            ref={textareaRef}
            className="composer-textarea"
            placeholder="Type a message..."
            rows={1}
            value={input}
            onChange={(e) => { setInput(e.target.value); autoResize(); }}
            onKeyDown={handleKeyDown}
          />
          <button
            className="btn-send"
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || loading}
          >
            {loading ? <span className="spinner" style={{ width: "16px", height: "16px", borderWidth: "2px" }} /> : <SendIcon />}
          </button>
        </div>
      </div>
    </div>
  );
}
