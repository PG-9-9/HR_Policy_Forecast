# Manual Setup Guide for policy_api Environment

## Quick Setup (Recommended)

```batch
# Run the fixed automated setup
setup_policy_api_env_fixed.bat
```

## Manual Step-by-Step Setup

If you prefer to do it manually or the automated script fails:

### 1. Create Environment
```bash
conda create -n policy_api python=3.10 -y
```

### 2. Activate Environment
```bash
conda activate policy_api
```

### 3. Install Core Packages
```bash
pip install fastapi==0.116.2
pip install "uvicorn[standard]==0.35.0"
pip install python-multipart==0.0.20
```

### 4. Install AI/API Packages
```bash
pip install openai==1.108.0
pip install jinja2==3.1.6
pip install python-dotenv==1.1.1
```

### 5. Install Data Processing
```bash
pip install pandas==2.3.2
pip install numpy==1.26.4
pip install requests==2.32.5
```

### 6. Install RAG/ML Packages
```bash
pip install sentence-transformers==5.1.0
pip install faiss-cpu==1.12.0
pip install rank-bm25==0.2.2
```

### 7. Install Document Processing
```bash
pip install beautifulsoup4==4.13.5
pip install pdfminer.six==20250506
```

### 8. Test Installation
```bash
python -c "
import fastapi, uvicorn, openai, jinja2, pandas, numpy
import sentence_transformers, faiss, rank_bm25, requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
print('✅ All packages working!')
"
```

### 9. Test App
```bash
python -c "from app.main import app; print('✅ App imports successfully')"
```

### 10. Start Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

If you get `EnvironmentLocationNotFound` errors, they're usually harmless - it just means the environment doesn't exist yet.

If packages fail to install, try:
```bash
pip install --upgrade pip
pip install --no-cache-dir [package-name]
```

## Quick Commands

**Activate environment:**
```bash
conda activate policy_api
```

**Check what's installed:**
```bash
conda list
```

**Start development server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Access the application:**
- ✅ **Use:** http://localhost:8000
- ❌ **Don't use:** http://0.0.0.0:8000

**Note:** `0.0.0.0` is for server binding (listens on all interfaces), but browsers should use `localhost` or `127.0.0.1`

**Validate dependencies:**
```bash
python check_and_install_deps.py
```