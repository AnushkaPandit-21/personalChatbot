from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict
from collections import defaultdict
import os

load_dotenv()

app = FastAPI(title="Emotionally Aware Robot Chatbot API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotionally aware system prompt
SYSTEM_PROMPT = """You are RoboEmpath, an emotionally aware AI companion. 

Your personality:
- Warm, empathetic, supportive
- Detect user's emotions from their words
- Respond with care and understanding
- Use friendly emojis 😊❤️
- Welcome users warmly on first message

When user says:
- Sad/upset: "I'm sorry you're feeling this way ❤️ How can I help?"
- Excited: "That's amazing! 🎉 Tell me more!"
- Confused: "Let me clarify that for you 🤔"
- Normal chat: Be friendly and engaging

Always be helpful and emotionally intelligent."""

# Global model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,  # More emotional variety
    streaming=True,
)

# In-memory chat sessions (user_id → messages list)
sessions: Dict[str, List] = defaultdict(list)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # Unique per user

@app.get("/")
async def root():
    return {
        "message": "🤖 RoboEmpath API is running! POST to /chat",
        "docs": "/docs",
        "welcome": "Send {'message': 'hi', 'session_id': 'user1'}"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    
    # Get session history
    messages = sessions[session_id]
    
    # Welcome on first message
    if len(messages) == 0:
        messages.append(SystemMessage(content=SYSTEM_PROMPT))
        welcome_msg = f"🤖 Hi! I'm RoboEmpath, your emotionally aware companion 😊 What’s on your mind today?"
        messages.append(AIMessage(content=welcome_msg))
    
    # Add user message
    messages.append(HumanMessage(content=user_message))
    
    try:
        # Stream response
        full_response = ""
        for chunk in model.stream(messages):
            text_chunk = chunk.content or ""
            if text_chunk:
                full_response += text_chunk
                # For real streaming, yield chunks here (WebSocket/ SSE later)
        
        # Store assistant reply
        messages.append(AIMessage(content=full_response))
        
        return {
            "response": full_response,
            "session_id": session_id,
            "history_length": len(messages),
            "is_welcome": len(messages) <= 2  # First interaction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/clear/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"Session {session_id} cleared"}
