import React, { useState, useRef, useEffect } from "react";
import ParticleCanvas from "./components/ParticleCanvas";
import WelcomePage from "./pages/WelcomePage";
import ModelSelectModal from "./components/ModelSelectModal";
import ChatPage from "./pages/ChatPage";

/**
 * Manages page transitions with slide+fade animations.
 * direction: "forward" (right→) | "back" (←left)
 */
export default function App() {
  const [view, setView] = useState("welcome");   // "welcome" | "modal" | "chat"
  const [selectedModel, setSelectedModel] = useState(null);
  const [animClass, setAnimClass] = useState("page-enter");
  const [isTransitioning, setIsTransitioning] = useState(false);
  const exitTimeout = useRef(null);

  /** Navigate forward (slide left-to-right content) */
  const navigateTo = (nextView, model = null) => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    setAnimClass("page-exit");

    exitTimeout.current = setTimeout(() => {
      if (model) setSelectedModel(model);
      setView(nextView);
      setAnimClass("page-enter");
      setIsTransitioning(false);
    }, 300); // matches --dur-page (350ms), slightly earlier for snappier feel
  };

  /** Navigate back (slide right-to-left content) */
  const navigateBack = (nextView) => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    setAnimClass("page-back-exit");

    exitTimeout.current = setTimeout(() => {
      setSelectedModel(null);
      setView(nextView);
      setAnimClass("page-back-enter");
      setIsTransitioning(false);
    }, 300);
  };

  useEffect(() => {
    return () => clearTimeout(exitTimeout.current);
  }, []);

  const pageStyle = { position: "relative", zIndex: 1, width: "100%" };

  return (
    <>
      <ParticleCanvas />
      <div className={animClass} style={pageStyle}>
        {view === "welcome" && (
          <WelcomePage onStart={() => navigateTo("modal")} />
        )}
        {view === "modal" && (
          <ModelSelectModal
            onSelect={(m) => navigateTo("chat", m)}
          />
        )}
        {view === "chat" && (
          <ChatPage
            selectedModel={selectedModel}
            onBack={() => navigateBack("modal")}
          />
        )}
      </div>
    </>
  );
}
