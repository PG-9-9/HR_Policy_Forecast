# app/main_minimal.py - Ultra-minimal version without RAG
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
# Remove RAG import for minimal build
# from rag.retrieve import stitched_context
from app.db import init, append, history, clear
from app.llm import chat_minimal, SYSTEM_MINIMAL

app = FastAPI(title="UK HR Policy Copilot - Minimal")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# Change Jinja2 template syntax to avoid conflicts with Vue.js
templates.env.variable_start_string = '{['
templates.env.variable_end_string = ']}'

# Initialize the database
init()

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and AWS"""
    return {"status": "healthy", "service": "UK HR Policy Copilot - Minimal"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint - OpenAI only (no local RAG)
    """
    try:
        question = chat_request.question.strip()
        conversation_id = chat_request.conversation_id or "default"
        
        if not question:
            return ChatResponse(
                response="Please provide a question about UK immigration policy or workforce planning.",
                conversation_id=conversation_id
            )
        
        # Get chat history
        messages = history(conversation_id)
        
        # For minimal version, just use OpenAI without RAG context
        response = chat_minimal(question, messages)
        
        # Store the conversation
        append(conversation_id, "user", question)
        append(conversation_id, "assistant", response)
        
        return ChatResponse(response=response, conversation_id=conversation_id)
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists."
        return ChatResponse(response=error_msg, conversation_id=conversation_id)

@app.post("/api/clear")
async def clear_history(conversation_id: str = "default"):
    """Clear conversation history"""
    clear(conversation_id)
    return {"message": "Conversation history cleared", "conversation_id": conversation_id}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    messages = history(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}

@app.get("/api/status")
async def status():
    """API status endpoint"""
    return {
        "status": "operational",
        "version": "minimal",
        "features": {
            "rag": False,
            "forecasting": False,
            "openai_chat": True
        }
    }

# Chat interface endpoint (form-based for simple testing)
@app.post("/chat", response_class=HTMLResponse)
async def chat_form(request: Request, message: str = Form(...)):
    """Form-based chat interface for testing"""
    try:
        messages = history("web")
        response = chat_minimal(message, messages)
        
        append("web", "user", message)
        append("web", "assistant", response)
        
        # Get updated history
        conversation = history("web")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "messages": conversation,
            "latest_response": response
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Sorry, I encountered an error. Please try again."
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)