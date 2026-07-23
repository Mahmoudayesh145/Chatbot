import React from "react";
import "./FluidBg.css";

export default function FluidBg() {
  return (
    <div className="fluid-bg-container">
      <div className="fluid-bg-gradient fluid-bg-gradient-1"></div>
      <div className="fluid-bg-gradient fluid-bg-gradient-2"></div>
      <div className="fluid-bg-gradient fluid-bg-gradient-3"></div>
      <div className="fluid-bg-gradient fluid-bg-gradient-4"></div>
      <div className="fluid-bg-noise"></div>
    </div>
  );
}
