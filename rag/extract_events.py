# rag/extract_events.py
import os
import json
import re
import calendar
from datetime import datetime, date
from typing import List, Dict, Iterable, Tuple

import pandas as pd
from openai import OpenAI

from .retrieve import search

# -----------------------------
# OpenAI client / model config
# -----------------------------
from dotenv import load_dotenv   

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------
# Event extraction prompt/schema
# -----------------------------
SCHEMA_INSTRUCTIONS = """You extract structured POLICY EVENTS that can impact hiring.

Return a JSON object with a top-level key "events", whose value is an array of objects
with the exact fields below:

- date (string): ISO-like date. If the passage provides a full date, use "YYYY-MM-DD".
  If only month/year are available, use "YYYY-MM-01".
- type (string): one of {RULE_CHANGE, THRESHOLD_UPDATE, ENFORCEMENT, WHITE_PAPER, COURT_RULING, ADMIN_NOTICE}
- topic (string): short topic label (e.g., "Skilled Worker salary threshold", "Sponsor licence enforcement").
- jurisdiction (string): country/region (e.g., "UK", "England", "UK-wide").
- summary (string): <= 60 words, neutral.
- impact (string): one of {POSITIVE, NEGATIVE, UNCLEAR} (impact on hiring ability/speed).
- confidence (number): 0..1 (confidence level).
- citations (array): each element is an object { "doc_id": "<doc_id from context>" }.
  Use only doc_ids present in the provided context. Do NOT invent or fetch others.

Constraints:
- Output STRICT JSON. No commentary. No markdown. No trailing commas.
- Only use information present in the provided context passages.
- If no concrete event found, return {"events": []}.
"""

USER_TMPL = """Analysis task: Extract HR/legal policy events from context passages with [doc_id] headers.

{context}

Now, extract events as specified:

{schema}
"""

# -----------------------------
# Robust date parsing utilities
# -----------------------------

_MONTH_ALIASES = {
    # common variations the LLM might emit
    "JAN": 1, "JANUARY": 1,
    "FEB": 2, "FEBRUARY": 2,
    "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MAY": 5,
    "JUN": 6, "JUNE": 6,
    "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12,
}

def _try_parse_iso(d: str) -> date | None:
    """Try strict ISO-like formats."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m", "%Y/%m", "%Y.%m"):
        try:
            dt = datetime.strptime(d, fmt)
            # If only year-month, normalize to first day
            if fmt in ("%Y-%m", "%Y/%m", "%Y.%m"):
                return date(dt.year, dt.month, 1)
            return dt.date()
        except Exception:
            pass
    return None

def _try_parse_loose(d: str) -> date | None:
    """Handle strings like '2024 OCT', 'OCT 2024', 'October 2024' → first of month."""
    s = re.sub(r"\s+", " ", d.strip().upper())
    # e.g., "2024 OCT", "2024 OCTOBER"
    m1 = re.match(r"^(20\d{2}|19\d{2})\s+([A-Z]+)$", s)
    if m1:
        year = int(m1.group(1))
        mon_name = m1.group(2)
        if mon_name in _MONTH_ALIASES:
            return date(year, _MONTH_ALIASES[mon_name], 1)

    # e.g., "OCT 2024", "OCTOBER 2024"
    m2 = re.match(r"^([A-Z]+)\s+(20\d{2}|19\d{2})$", s)
    if m2:
        mon_name = m2.group(1)
        year = int(m2.group(2))
        if mon_name in _MONTH_ALIASES:
            return date(year, _MONTH_ALIASES[mon_name], 1)

    # e.g., "2024-10-??" where day missing or "Oct-2024"
    try:
        # dateutil-like fallback via pandas
        dt = pd.to_datetime(d, errors="coerce")
        if pd.isna(dt):
            return None
        # normalize to first of month if day==NaT or ambiguous
        return date(dt.year, dt.month, 1)
    except Exception:
        return None

def parse_event_date(s: str) -> date | None:
    """Parse event date string to a date; if month only, returns first-of-month."""
    if not s or not isinstance(s, str):
        return None
    return _try_parse_iso(s) or _try_parse_loose(s)

def month_label(dt_obj: date) -> str:
    """Return label like 'YYYY MON' (e.g., '2001 MAY') matching the CSV format."""
    return f"{dt_obj.year} {calendar.month_abbr[dt_obj.month].upper()}"

# -----------------------------
# Impact mapping
# -----------------------------

_IMPACT_MAP = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "UNCLEAR": 0.0}

def impact_score(impact: str, confidence: float) -> float:
    return float(_IMPACT_MAP.get(impact.upper(), 0.0) * max(0.0, min(1.0, confidence)))

# -----------------------------
# Core: extract events with OpenAI
# -----------------------------

def extract_events_for_query(query: str, k: int = 8) -> List[Dict]:
    """
    Retrieve context via hybrid search, then ask OpenAI for strictly structured events.
    Returns a list of event dicts with fields:
      date, type, topic, jurisdiction, summary, impact, confidence, citations
    Dates are normalized to 'YYYY-MM-DD' or 'YYYY-MM-01' in the result.
    """
    hits = search(query, k=k)
    if not hits:
        return []

    # Stitch context (with [doc_id])
    ctx = ""
    for h in hits:
        ctx += f"[{h['doc_id']}] {h['passage']}\n\n"

    prompt = USER_TMPL.format(context=ctx, schema=SCHEMA_INSTRUCTIONS)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Extract precise, citable policy events from the provided passages."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception as e:
        # If the model emits a raw array instead of {"events":[...]} handle gracefully
        try:
            events = json.loads(raw)
            if isinstance(events, list):
                data = {"events": events}
            else:
                raise e
        except Exception:
            raise ValueError(f"OpenAI did not return valid JSON. Got: {raw[:300]} ...") from e

    events = data.get("events", [])
    normalized: List[Dict] = []
    for ev in events:
        d = parse_event_date(ev.get("date", ""))
        if d is None:
            # If we cannot parse any date, skip this event (cannot align to months)
            continue
        ev_norm = {
            "date": d.isoformat(),  # normalized
            "type": ev.get("type", "").strip(),
            "topic": ev.get("topic", "").strip(),
            "jurisdiction": ev.get("jurisdiction", "").strip(),
            "summary": ev.get("summary", "").strip(),
            "impact": ev.get("impact", "").strip().upper(),
            "confidence": float(ev.get("confidence", 0.5)),
            "citations": ev.get("citations", []),
        }
        normalized.append(ev_norm)

    return normalized

# -----------------------------
# Alignment: events -> monthly features
# -----------------------------

def monthly_features_for_labels(
    month_labels: Iterable[str],
    events: List[Dict],
) -> pd.DataFrame:
    """
    Convert events to features aligned with a sequence of month labels like
    ['2001 MAY', '2001 JUN', ...] (ONS CSV format).

    Returns a DataFrame with columns: ['month_label', 'event_flag', 'event_impact']
    where:
      - event_flag = 1.0 if any event occurs in that month, else 0.0
      - event_impact = sum over events in that month of impact_score(impact, confidence)
    """
    # Pre-index events by month_label
    by_label: Dict[str, List[Dict]] = {}
    for ev in events:
        d = parse_event_date(ev.get("date", ""))
        if not d:
            continue
        lab = month_label(d)  # e.g., "2024 OCT"
        by_label.setdefault(lab, []).append(ev)

    rows = []
    for lab in month_labels:
        evs = by_label.get(lab, [])
        if evs:
            flag = 1.0
            imp = sum(impact_score(e.get("impact", "UNCLEAR"), float(e.get("confidence", 0.5))) for e in evs)
        else:
            flag = 0.0
            imp = 0.0
        rows.append({"month_label": lab, "event_flag": flag, "event_impact": float(imp)})
    return pd.DataFrame(rows)

def monthly_features_for_dates(
    months: Iterable[pd.Timestamp],
    events: List[Dict],
) -> pd.DataFrame:
    """
    Same as above but takes pandas Timestamps (first-of-month recommended) and returns:
    ['date','event_flag','event_impact'] with 'date' as Timestamp.
    """
    labels = [month_label(d.date()) for d in months]
    df = monthly_features_for_labels(labels, events)
    out = pd.DataFrame({"date": list(months)})
    out = out.merge(df.rename(columns={"month_label": "label"}), left_index=True, right_index=True, how="left")
    out = out.drop(columns=["label"]).fillna({"event_flag": 0.0, "event_impact": 0.0})
    return out

# -----------------------------
# Convenience: end-to-end helper
# -----------------------------

def extract_and_align_for_series_labels(query: str, series_month_labels: Iterable[str], k: int = 8) -> Tuple[List[Dict], pd.DataFrame]:
    """
    One call to:
      - retrieve + extract events,
      - align them to a series of 'YYYY MON' labels coming from the ONS CSV.

    Returns: (events_list, monthly_features_df)
    """
    events = extract_events_for_query(query, k=k)
    feats = monthly_features_for_labels(series_month_labels, events)
    return events, feats

# -----------------------------
# CLI smoke test (optional)
# -----------------------------
