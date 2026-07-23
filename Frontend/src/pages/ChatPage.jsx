import React from "react";
import GeminiChat from "../components/GeminiChat";
import SinglePrompt from "../components/SinglePrompt";
import TranslationPanel from "../components/TranslationPanel";

const BackIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 12H5M12 19l-7-7 7-7"/>
  </svg>
);

export default function ChatPage({ selectedModel, onBack }) {
  if (!selectedModel) return null;

  const renderModel = () => {
    if (selectedModel.id === "gemini") return <GeminiChat model={selectedModel} />;
    if (selectedModel.id.startsWith("translate")) return <TranslationPanel model={selectedModel} />;
    return <SinglePrompt model={selectedModel} />;
  };

  return (
    <div className="app-shell">
      {/* Top Bar */}
      <div className="app-topbar">
        <div className="topbar-left">
          <button className="btn-back" onClick={onBack}>
            <BackIcon /> Models
          </button>
          <div className="topbar-breadcrumb">
            <span className="topbar-breadcrumb-sep">/</span>
            <span className="topbar-breadcrumb-current">{selectedModel.name}</span>
          </div>
        </div>

        <div className="topbar-right">
          <div className="model-tag-top">{selectedModel.tag}</div>
          <div className="status-badge">
            <span className="status-badge-dot" />
            Live
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="app-content">
        {renderModel()}
      </div>
    </div>
  );
}
