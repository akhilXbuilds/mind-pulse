import { useRef, useState } from "react";
import {
  Activity,
  Brain,
  Camera,
  Heart,
  MessageCircle,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState(
    "Hi. I'm here to listen. How are you feeling today?"
  );

  const videoRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const enableCamera = async () => {
    try {
      setCameraError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraActive(true);
    } catch (error) {
      console.error("Camera error:", error);
      setCameraError(
        "Camera access was denied or the camera is unavailable."
      );
    }
  };

  const handleSend = () => {
    if (!message.trim()) {
      return;
    }

    const userMessage = message.trim().toLowerCase();

    if (
      userMessage.includes("suicide") ||
      userMessage.includes("kill myself") ||
      userMessage.includes("end my life") ||
      userMessage.includes("want to die")
    ) {
      setResponse(
        "I'm really sorry you're going through this. You deserve immediate human support. Please contact a trusted person or your local emergency/crisis service right now."
      );
    } else if (
      userMessage.includes("sad") ||
      userMessage.includes("depressed") ||
      userMessage.includes("lonely") ||
      userMessage.includes("bad")
    ) {
      setResponse(
        "I'm here with you. You don't have to figure everything out at once. Try taking a slow breath and telling me what has been weighing on you."
      );
    } else if (
      userMessage.includes("stress") ||
      userMessage.includes("stressed") ||
      userMessage.includes("anxious") ||
      userMessage.includes("anxiety")
    ) {
      setResponse(
        "That sounds difficult. Let's slow things down for a moment. Take a few calm breaths and focus on one thing you can control right now."
      );
    } else if (
      userMessage.includes("happy") ||
      userMessage.includes("good") ||
      userMessage.includes("great")
    ) {
      setResponse(
        "That's good to hear. Keep that momentum going. What's making you feel good today?"
      );
    } else {
      setResponse(
        "Thank you for sharing that with me. I'm listening. Tell me a little more about how you're feeling."
      );
    }

    setMessage("");
  };

  return (
    <div className="app">
      {/* NAVBAR */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Brain size={24} />
          </div>

          <div>
            <h1>MindGuard AI</h1>
            <span>Smart Mental Wellness Assistant</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          System Online
        </div>
      </header>

      <main className="dashboard">
        {/* HERO */}
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow">
              <Sparkles size={15} />
              AI-POWERED WELLNESS
            </div>

            <h2>
              Your mind matters.
              <br />
              <span>We're here to listen.</span>
            </h2>

            <p className="hero-text">
              A private AI assistant that helps you understand your
              emotional state and provides supportive guidance when you
              need it.
            </p>
          </div>

          <div className="hero-icon">
            <Heart size={70} />
          </div>
        </section>

        {/* MAIN GRID */}
        <section className="grid">
          {/* CAMERA CARD */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-label">EMOTIONAL ANALYSIS</div>
                <h3>Facial Expression</h3>
              </div>

              <Camera size={23} />
            </div>

            <div className="camera-placeholder">
              {!cameraActive ? (
                <>
                  <Camera size={52} />

                  <p>Camera not connected</p>

                  <span>
                    Enable camera to analyze facial expressions
                  </span>

                  <button
                    className="primary-btn"
                    onClick={enableCamera}
                  >
                    Enable Camera
                  </button>

                  {cameraError && (
                    <span className="camera-error">
                      {cameraError}
                    </span>
                  )}
                </>
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="camera-video"
                />
              )}
            </div>
          </div>

          {/* EMOTION CARD */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-label">CURRENT STATE</div>
                <h3>Emotional Overview</h3>
              </div>

              <Activity size={23} />
            </div>

            <div className="emotion">
              <div className="emotion-circle">😐</div>

              <div>
                <h4>Neutral</h4>
                <p>Confidence: 82%</p>
              </div>
            </div>

            <div className="risk">
              <div>
                <span>Risk Level</span>
                <strong>LOW</strong>
              </div>

              <ShieldCheck size={26} />
            </div>
          </div>
        </section>

        {/* CHAT */}
        <section className="card chat-card">
          <div className="card-header">
            <div>
              <div className="card-label">AI SUPPORT</div>
              <h3>Wellness Assistant</h3>
            </div>

            <MessageCircle size={23} />
          </div>

          <div className="assistant-message">
            <div className="assistant-avatar">
              <Brain size={20} />
            </div>

            <p>{response}</p>
          </div>

          <div className="chat-input">
            <input
              type="text"
              placeholder="Tell me how you're feeling..."
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleSend();
                }
              }}
            />

            <button onClick={handleSend}>
              <Send size={18} />
            </button>
          </div>
        </section>

        <p className="disclaimer">
          MindGuard AI is a supportive prototype and does not provide
          medical diagnosis or replace professional mental-health care.
        </p>
      </main>
    </div>
  );
}

export default App;