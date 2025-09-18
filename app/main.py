# app/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
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
    
    # Check if query is relevant to UK immigration/workforce topics
    relevant_keywords = [
        'visa', 'immigration', 'sponsor', 'licence', 'skilled worker', 'work permit',
        'job', 'vacancy', 'workforce', 'employment', 'recruitment', 'hr', 'policy',
        'forecast', 'trend', 'market', 'uk', 'britain', 'british', 'government',
        'ons', 'salary', 'threshold', 'points', 'compliance', 'employer'
    ]
    
    msg_lower = msg.lower()
    is_relevant = any(keyword in msg_lower for keyword in relevant_keywords)
    
    if not is_relevant:
        return "I specialize in UK immigration policy and workforce planning. Please ask questions about UK visa requirements, immigration rules, sponsor licences, job market trends, or workforce forecasting."
    
    # RAG context for this turn
    ctx = stitched_context(msg, k=6)
    
    # Check if RAG found relevant context
    if not ctx.strip():
        return "I don't have current information about that specific topic. Please refer to the latest guidance on gov.uk or consult with an immigration specialist."
    
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
    
    return reply

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
