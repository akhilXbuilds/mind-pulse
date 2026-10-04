import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Make sure your backend/.env file contains GEMINI_API_KEY."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Mind Pulse API",
    description="AI-powered emotional wellness assistant",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):
    reply: str
    emotion: str
    risk: str
    confidence: int


# ============================================================
# CONVERSATION MEMORY
# ============================================================

# Example:
#
# conversations = {
#     "default": [
#         {
#             "role": "user",
#             "text": "I'm stressed about exams"
#         },
#         {
#             "role": "model",
#             "text": "What part of the exams is worrying you?"
#         }
#     ]
# }

conversations = {}


# ============================================================
# MIND PULSE PERSONALITY
# ============================================================

SYSTEM_INSTRUCTION = """
You are Mind Pulse, an AI-powered emotional wellness assistant.

Your purpose is to provide supportive, natural and personalized
conversations that help users reflect on what they are experiencing.

PERSONALITY:
- Calm
- Warm
- Empathetic
- Intelligent
- Friendly
- Non-judgmental
- Natural
- Conversational

CONVERSATION RULES:

1. Understand the user's actual message before responding.

2. Give responses specific to what the user said.

3. Do NOT give the same response repeatedly.

4. Do NOT constantly say:
   - "Take a deep breath."
   - "Everything will be okay."
   - "I'm sorry you're feeling this way."

5. Avoid generic motivational speeches unless they are actually useful.

6. Ask a follow-up question when it naturally helps the conversation.

7. Remember information from earlier messages in the conversation.

8. If the user says something related to exams, college, relationships,
   work, loneliness, stress, anxiety, anger, confidence, etc.,
   respond specifically to that situation.

9. Keep normal responses relatively concise:
   usually around 2-5 sentences.

10. Speak naturally, like a thoughtful AI companion.

11. Never pretend to be human.

12. Never claim to diagnose a mental health condition.

13. Never claim that facial expressions, text or AI can medically
    diagnose someone's mental health.

14. If the user is in immediate danger or expresses an intention
    to seriously hurt themselves or someone else, prioritize
    immediate real-world help.

15. In a crisis, encourage the user to contact local emergency
    services, a crisis service, or a trusted person who can
    stay with them.

16. Never provide instructions for self-harm, suicide, violence,
    or other dangerous activities.

Mind Pulse is an emotional wellness support tool, not a medical
diagnosis system.
"""


# ============================================================
# EMOTION DETECTION
# ============================================================

def detect_emotion(message: str) -> tuple[str, int]:

    text = message.lower()

    # --------------------------------------------------------
    # Distressed
    # --------------------------------------------------------

    distressed_words = [
        "suicide",
        "kill myself",
        "end my life",
        "want to die",
        "i want to die",
        "hurt myself",
        "harm myself",
        "self harm",
        "self-harm",
        "no reason to live",
        "can't go on",
        "cannot go on",
    ]

    for word in distressed_words:

        if word in text:
            return "Distressed", 96

    # --------------------------------------------------------
    # Angry
    # --------------------------------------------------------

    angry_words = [
        "angry",
        "furious",
        "rage",
        "mad",
        "annoyed",
        "pissed",
        "frustrated",
        "hate",
    ]

    for word in angry_words:

        if word in text:
            return "Angry", 84

    # --------------------------------------------------------
    # Sad
    # --------------------------------------------------------

    sad_words = [
        "sad",
        "crying",
        "cry",
        "lonely",
        "heartbroken",
        "hopeless",
        "upset",
        "unhappy",
    ]

    for word in sad_words:

        if word in text:
            return "Sad", 84

    # --------------------------------------------------------
    # Stressed
    # --------------------------------------------------------

    stressed_words = [
        "stress",
        "stressed",
        "exam",
        "exams",
        "pressure",
        "overwhelmed",
        "deadline",
        "anxious",
        "anxiety",
        "worried",
        "worry",
        "tension",
        "nervous",
    ]

    for word in stressed_words:

        if word in text:
            return "Stressed", 84

    # --------------------------------------------------------
    # Happy
    # --------------------------------------------------------

    happy_words = [
        "happy",
        "great",
        "excited",
        "awesome",
        "amazing",
        "joy",
        "glad",
        "good",
        "wonderful",
        "proud",
    ]

    for word in happy_words:

        if word in text:
            return "Happy", 88

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "Neutral", 78


# ============================================================
# RISK DETECTION
# ============================================================

def detect_risk(message: str) -> str:

    text = message.lower()

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    high_risk_words = [
        "kill myself",
        "suicide",
        "commit suicide",
        "end my life",
        "want to die",
        "i want to die",
        "going to kill myself",
        "hurt myself",
        "harm myself",
        "self harm",
        "self-harm",
        "no reason to live",
        "can't go on",
        "cannot go on",
    ]

    for word in high_risk_words:

        if word in text:
            return "HIGH"

    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    medium_risk_words = [
        "hopeless",
        "worthless",
        "helpless",
        "completely alone",
        "can't cope",
        "cannot cope",
        "falling apart",
        "extremely depressed",
    ]

    for word in medium_risk_words:

        if word in text:
            return "MEDIUM"

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# CONVERSATION FUNCTIONS
# ============================================================

def save_message(
    session_id: str,
    role: str,
    text: str
):

    if session_id not in conversations:

        conversations[session_id] = []

    conversations[session_id].append(
        {
            "role": role,
            "text": text
        }
    )

    # Keep only the latest 20 messages.
    conversations[session_id] = conversations[session_id][-20:]


def build_gemini_history(session_id: str):

    history = conversations.get(
        session_id,
        []
    )

    contents = []

    for message in history:

        contents.append(
            types.Content(
                role=message["role"],
                parts=[
                    types.Part(
                        text=message["text"]
                    )
                ]
            )
        )

    return contents


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Mind Pulse API is running",
        "status": "online",
        "model": MODEL_NAME
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "gemini": "configured",
        "application": "Mind Pulse"
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    message = request.message.strip()

    if not message:

        return ChatResponse(
            reply="I'm here. Tell me what's on your mind.",
            emotion="Neutral",
            risk="LOW",
            confidence=78
        )

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    session_id = request.session_id or "default"

    # --------------------------------------------------------
    # Emotion
    # --------------------------------------------------------

    emotion, confidence = detect_emotion(
        message
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk = detect_risk(
        message
    )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    save_message(
        session_id=session_id,
        role="user",
        text=message
    )

    # --------------------------------------------------------
    # Get conversation history
    # --------------------------------------------------------

    history = build_gemini_history(
        session_id
    )

    # --------------------------------------------------------
    # Dynamic system context
    # --------------------------------------------------------

    dynamic_instruction = f"""
CURRENT MESSAGE ANALYSIS

Estimated emotion:
{emotion}

Estimated safety risk:
{risk}

These are simple application-level signals.
They are NOT medical diagnoses.

Use this information only as supporting context.
The user's actual words and conversation context
are more important than these labels.

Respond naturally to the user.
"""

    # --------------------------------------------------------
    # Gemini generation
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=(
                    SYSTEM_INSTRUCTION
                    + "\n\n"
                    + dynamic_instruction
                ),
                temperature=0.9,
                max_output_tokens=500
            )
        )

        reply = response.text.strip()

        if not reply:

            reply = (
                "I'm listening. Tell me a little more "
                "about what's going on."
            )

    except Exception as error:

        print(
            "Gemini API Error:",
            error
        )

        reply = (
            "I'm having trouble connecting to my AI service "
            "right now. Please try sending that again."
        )

    # --------------------------------------------------------
    # Save AI response
    # --------------------------------------------------------

    save_message(
        session_id=session_id,
        role="model",
        text=reply
    )

    # --------------------------------------------------------
    # HIGH-RISK SAFETY RESPONSE
    # --------------------------------------------------------

    if risk == "HIGH":

        reply += (
            "\n\nIf you feel you may act on these thoughts "
            "or you're in immediate danger, please contact "
            "your local emergency service or a crisis service "
            "right now, and reach out to someone you trust "
            "who can stay with you."
        )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return ChatResponse(
        reply=reply,
        emotion=emotion,
        risk=risk,
        confidence=confidence
    )


# ============================================================
# CLEAR CHAT
# ============================================================

@app.delete(
    "/chat/{session_id}"
)
def clear_chat(session_id: str):

    if session_id in conversations:

        del conversations[session_id]

    return {
        "message": "Conversation cleared",
        "session_id": session_id
    }