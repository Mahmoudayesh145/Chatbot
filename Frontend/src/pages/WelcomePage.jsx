import React from "react";

export default function WelcomePage({ onStart }) {
  return (
    <div className="welcome-page">
      <div className="welcome-inner">

        <div className="welcome-tag anim-fade-up delay-0">
          <span className="welcome-tag-dot" />
          Unified AI Platform
        </div>

        <h1 className="welcome-title anim-fade-up delay-1">
          Machine Intelligence,
          <br />
          <span className="welcome-title-grad">Redefined.</span>
        </h1>

        <p className="welcome-sub anim-fade-up delay-2">
          A bespoke interface for production-grade NLP models — from real-time
          multi-turn dialogue to precision language translation and text analysis.
        </p>

        <div className="anim-fade-up delay-3">
          <button className="btn-cta" onClick={onStart}>
            Explore Models
            <svg className="btn-cta-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </button>
        </div>

        <div className="welcome-stats anim-fade-up delay-4">
          <div className="welcome-stat">
            <span className="welcome-stat-num">7</span>
            <span className="welcome-stat-label">Models</span>
          </div>
          <div className="welcome-stat-divider" />
          <div className="welcome-stat">
            <span className="welcome-stat-num">4</span>
            <span className="welcome-stat-label">Languages</span>
          </div>
          <div className="welcome-stat-divider" />
          <div className="welcome-stat">
            <span className="welcome-stat-num">Real-Time</span>
            <span className="welcome-stat-label">Inference</span>
          </div>
        </div>

      </div>
    </div>
  );
}
