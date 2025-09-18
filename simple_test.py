#!/usr/bin/env python3

import sys
print("Python path:", sys.executable)

try:
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except ImportError as e:
    print("✗ FastAPI import failed:", e)

try:
    from app.db import init
    init()
    print("✓ Database initialized successfully")
except Exception as e:
    print("✗ Database initialization failed:", e)

try:
    from app.llm import SYSTEM
    print("✓ LLM module imported successfully")
    print("System prompt length:", len(SYSTEM))
except Exception as e:
    print("✗ LLM import failed:", e)

try:
    from rag.retrieve import stitched_context
    print("✓ RAG module imported successfully")
except Exception as e:
    print("✗ RAG import failed:", e)

print("All basic imports completed.")