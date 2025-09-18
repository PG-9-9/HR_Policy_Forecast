# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rag.retrieve import stitched_context
from forecasting.infer import forecast_months
from app.db import init, append, history, clear
from app.llm import chat, SYSTEM

app = FastAPI(title="UK HR Policy Copilot + Forecast")

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

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["POST /chat", "GET /forecast?h=6"]}
