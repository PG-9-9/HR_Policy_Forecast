# app/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rag.retrieve import stitched_context
from forecasting.infer import forecast_months
from app.db import init, append, history, clear
from app.llm import chat, SYSTEM

app = FastAPI(title="UK HR Policy Copilot + Forecast")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# Change Jinja2 template syntax to avoid conflicts with Vue.js
templates.env.variable_start_string = '{['
templates.env.variable_end_string = ']}'

class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 6
    reset: Optional[bool] = False

class ChatResponse(BaseModel):
    session_id: str
    reply: str

class ForecastResponse(BaseModel):
    horizon: int
    forecast: List[Dict[str, Any]]
    events: List[Dict[str, Any]]

@app.on_event("startup")
def _init():
    init()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get")
async def get_bot_response(msg: str = Form(...)):
    """Handle chat messages from the web interface"""
    session_id = "web_session"  # Use a default session for web interface
    
    # Check if query is relevant to UK immigration/workforce topics OR is a greeting
    relevant_keywords = [
        'visa', 'immigration', 'sponsor', 'licence', 'skilled worker', 'work permit',
        'job', 'vacancy', 'workforce', 'employment', 'recruitment', 'hr', 'policy',
        'forecast', 'trend', 'market', 'uk', 'britain', 'british', 'government',
        'ons', 'salary', 'threshold', 'points', 'compliance', 'employer'
    ]
    
    greeting_keywords = ['hi', 'hello', 'hey', 'help', 'what', 'how', 'who', 'assistant', 'can you']
    
    msg_lower = msg.lower()
    is_relevant = any(keyword in msg_lower for keyword in relevant_keywords)
    is_greeting = any(keyword in msg_lower for keyword in greeting_keywords)
    
    # If it's not relevant and not a greeting, give fallback response
    if not is_relevant and not is_greeting:
        fallback = "Hello! I'm your HR Assistant, specialized in UK immigration policy and workforce planning. I help HR professionals, hiring managers, and employers navigate UK immigration requirements and make data-driven recruitment decisions. I can provide immigration guidance, workforce forecasting, and policy impact analysis. How can I assist you today?"
        return PlainTextResponse(content=fallback, media_type="text/plain")
    
    # For greetings, use a clear structured response
    if is_greeting and not is_relevant:
        greeting_lines = [
            "Hello! I'm your HR Assistant. How can I assist you today?",
            "",
            "**Core Services:**",
            "",
            "1. **Workforce Forecasting:** UK job vacancy predictions, labour market trends",
            "",
            "2. **Policy Impact Analysis:** How immigration rule changes affect recruitment strategies",
            "",
            "3. **Immigration Guidance:** UK visa requirements, Skilled Worker visas, sponsor licence obligations",
            "",
            "**What you can forecast with me:**",
            "",
            "• UK Job Vacancy Ratio Forecasting (up to 6 months ahead)",
            "• Immigration Policy Impact Predictions on job markets",
            "• Workforce demand trends based on ONS employment data",
            "• Event-driven forecasting incorporating policy changes and immigration rule updates",
            "",
            "What specific topic would you like to explore?"
        ]
        greeting_response = "\n".join(greeting_lines)
        
        append(session_id, "user", msg)
        append(session_id, "assistant", greeting_response)
        
        return PlainTextResponse(content=greeting_response, media_type="text/plain")
    
    # RAG context for this turn (for immigration/workforce topics)
    ctx = stitched_context(msg, k=6)
    
    # Check if RAG found relevant context
    if not ctx.strip():
        no_context = "I don't have current information about that specific topic. Please refer to the latest guidance on gov.uk or consult with an immigration specialist."
        return PlainTextResponse(content=no_context, media_type="text/plain")
    
    # Build messages
    msgs = [{"role":"system","content": SYSTEM},
            {"role":"system","content": f"Context:\n{ctx}"}]
    
    # Add history (persisted)
    for r, c in history(session_id):
        msgs.append({"role":r, "content":c})
    msgs.append({"role":"user","content": msg})
    
    reply = chat(msgs)
    append(session_id, "user", msg)
    append(session_id, "assistant", reply)
    
    return PlainTextResponse(content=reply, media_type="text/plain")

@app.post("/chat", response_model=ChatResponse)
def chat_api(req: ChatRequest):
    if req.reset:
        clear(req.session_id)
    # RAG context for this turn
    ctx = stitched_context(req.message, k=req.top_k or 6)
    # Build messages
    msgs = [{"role":"system","content": SYSTEM},
            {"role":"system","content": f"Context:\n{ctx}"}]
    # history (persisted)
    for r,c in history(req.session_id):
        msgs.append({"role":r, "content":c})
    msgs.append({"role":"user","content": req.message})
    reply = chat(msgs)
    append(req.session_id, "user", req.message)
    append(req.session_id, "assistant", reply)
    return {"session_id": req.session_id, "reply": reply}

@app.get("/forecast", response_model=ForecastResponse)
def forecast_api(h: int = 6):
    out, events = forecast_months(h)
    return {"horizon": h, "forecast": out.to_dict(orient="records"), "events": events}
