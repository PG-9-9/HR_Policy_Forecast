# app/db.py
from pathlib import Path
import sqlite3
from typing import List, Tuple

DB = Path("data/app.sqlite")
DB.parent.mkdir(parents=True, exist_ok=True)

def init():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS memory(
        session_id TEXT, role TEXT, content TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit(); con.close()

def append(session_id: str, role: str, content: str):
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("INSERT INTO memory(session_id, role, content) VALUES (?,?,?)",
                (session_id, role, content))
    con.commit(); con.close()

def history(session_id: str, limit_pairs: int = 12) -> List[Tuple[str,str]]:
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("""SELECT role, content FROM memory
                   WHERE session_id=? ORDER BY ts DESC LIMIT ?""",
                (session_id, limit_pairs*2))
    rows = cur.fetchall()[::-1]  # chronological
    con.close()
    return rows

def clear(session_id: str):
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("DELETE FROM memory WHERE session_id=?", (session_id,))
    con.commit(); con.close()
