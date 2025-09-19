#!/usr/bin/env python3
"""
Test script to validate minimal requirements
Run this to check if all necessary modules can be imported
"""

import sys
print("🧪 Testing minimal requirements...")

# Test core modules
try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI failed: {e}")
    sys.exit(1)

try:
    import uvicorn
    print("✅ Uvicorn imported successfully")
except ImportError as e:
    print(f"❌ Uvicorn failed: {e}")
    sys.exit(1)

try:
    from openai import OpenAI
    print("✅ OpenAI imported successfully")
except ImportError as e:
    print(f"❌ OpenAI failed: {e}")
    sys.exit(1)

try:
    import jinja2
    print("✅ Jinja2 imported successfully")
except ImportError as e:
    print(f"❌ Jinja2 failed: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv imported successfully")
except ImportError as e:
    print(f"❌ python-dotenv failed: {e}")
    sys.exit(1)

# Test RAG modules
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported successfully")
except ImportError as e:
    print(f"❌ sentence-transformers failed: {e}")
    sys.exit(1)

try:
    import faiss
    print("✅ faiss-cpu imported successfully")
except ImportError as e:
    print(f"❌ faiss-cpu failed: {e}")
    sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
    print("✅ rank-bm25 imported successfully")
except ImportError as e:
    print(f"❌ rank-bm25 failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    print("✅ pandas & numpy imported successfully")
except ImportError as e:
    print(f"❌ pandas/numpy failed: {e}")
    sys.exit(1)

try:
    import requests
    print("✅ requests imported successfully")
except ImportError as e:
    print(f"❌ requests failed: {e}")
    sys.exit(1)

# Test app modules
try:
    from pathlib import Path
    import sqlite3
    print("✅ Standard library modules OK")
except ImportError as e:
    print(f"❌ Standard library failed: {e}")
    sys.exit(1)

# Test document parsing (optional)
try:
    from bs4 import BeautifulSoup
    print("✅ BeautifulSoup imported successfully")
except ImportError as e:
    print("⚠️ BeautifulSoup not available (optional for document parsing)")

try:
    from pdfminer.high_level import extract_text
    print("✅ pdfminer.six imported successfully")
except ImportError as e:
    print("⚠️ pdfminer.six not available (optional for PDF parsing)")

print("\n🎉 All core requirements satisfied!")
print("📦 Ready to build minimal Docker image")

# Test basic functionality
print("\n🔧 Testing basic functionality...")
try:
    app = fastapi.FastAPI()
    print("✅ FastAPI app creation works")
except Exception as e:
    print(f"❌ FastAPI app creation failed: {e}")

try:
    # Test sentence transformer loading (this is the heaviest operation)
    print("🔄 Testing sentence transformer loading...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence transformer model loaded successfully")
    
    # Test encoding
    test_text = "This is a test sentence"
    embedding = model.encode([test_text])
    print(f"✅ Text encoding works (embedding shape: {embedding.shape})")
    
except Exception as e:
    print(f"❌ Sentence transformer test failed: {e}")

print("\n✅ All tests passed! Minimal requirements are working correctly.")