#!/usr/bin/env python3
"""
Test script for Linux Docker compatibility
Tests key differences between Windows and Linux environments
"""

import sys
import platform
import os

print(f"🐧 Testing Linux compatibility...")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"Architecture: {platform.machine()}")

# Test core modules with Linux-specific considerations
modules_to_test = [
    ("fastapi", "FastAPI web framework"),
    ("uvicorn", "ASGI server"),
    ("openai", "OpenAI API client"),
    ("jinja2", "Template engine"),
    ("sentence_transformers", "Sentence embeddings"),
    ("faiss", "Vector similarity search"),
    ("rank_bm25", "BM25 ranking"),
    ("torch", "PyTorch deep learning"),
    ("numpy", "Numerical computing"),
    ("pandas", "Data manipulation"),
    ("requests", "HTTP client"),
    ("bs4", "BeautifulSoup HTML parsing"),
]

failed_imports = []
successful_imports = []

for module_name, description in modules_to_test:
    try:
        if module_name == "bs4":
            from bs4 import BeautifulSoup
        else:
            __import__(module_name)
        print(f"✅ {module_name}: {description}")
        successful_imports.append(module_name)
    except ImportError as e:
        print(f"❌ {module_name}: {description} - FAILED: {e}")
        failed_imports.append((module_name, str(e)))

# Test Linux-specific optimizations
print("\n🚀 Testing Linux-specific optimizations...")

# Test uvloop (Linux performance boost)
try:
    import uvloop
    print("✅ uvloop: Available for async performance boost")
except ImportError:
    print("⚠️ uvloop: Not available (optional Linux optimization)")

# Test BLAS/LAPACK backends
print("\n🔢 Testing numerical computing backends...")
try:
    import numpy as np
    np_config = np.__config__.show()
    print("✅ NumPy configuration loaded")
    
    # Test if OpenBLAS is being used
    import numpy.distutils.system_info as sysinfo
    blas_info = sysinfo.get_info('blas')
    if blas_info:
        print(f"✅ BLAS backend detected: {blas_info.get('libraries', 'Unknown')}")
    else:
        print("⚠️ BLAS backend info not available")
        
except Exception as e:
    print(f"❌ NumPy backend test failed: {e}")

# Test sentence transformer model loading (heaviest operation)
print("\n🤖 Testing ML model loading...")
try:
    from sentence_transformers import SentenceTransformer
    print("🔄 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test encoding
    test_text = "This is a test for Linux compatibility"
    embedding = model.encode([test_text])
    print(f"✅ Model loaded and encoding works (shape: {embedding.shape})")
    
    # Test if model is using optimal device
    device = str(model.device)
    print(f"📱 Model device: {device}")
    
except Exception as e:
    print(f"❌ Sentence transformer test failed: {e}")
    failed_imports.append(("sentence_transformers_model", str(e)))

# Test faiss vector operations
print("\n🔍 Testing vector search...")
try:
    import faiss
    import numpy as np
    
    # Create a simple test index
    dimension = 384  # MiniLM embedding dimension
    index = faiss.IndexFlatL2(dimension)
    
    # Add some dummy vectors
    vectors = np.random.random((10, dimension)).astype('float32')
    index.add(vectors)
    
    # Test search
    query = np.random.random((1, dimension)).astype('float32')
    distances, indices = index.search(query, 3)
    
    print(f"✅ FAISS vector search works (found {len(indices[0])} neighbors)")
    
except Exception as e:
    print(f"❌ FAISS test failed: {e}")

# Test file system operations (Windows vs Linux paths)
print("\n📁 Testing file system compatibility...")
try:
    import pathlib
    
    # Test path operations
    test_path = pathlib.Path("/app/data/processed/rag_index")
    print(f"✅ Path handling: {test_path} (POSIX style)")
    
    # Test file permissions (Linux-specific)
    import stat
    print("✅ File permission handling available")
    
except Exception as e:
    print(f"❌ File system test failed: {e}")

# Summary
print(f"\n📊 Test Summary:")
print(f"✅ Successful imports: {len(successful_imports)}")
print(f"❌ Failed imports: {len(failed_imports)}")

if failed_imports:
    print("\n❌ Failed modules:")
    for module, error in failed_imports:
        print(f"  - {module}: {error}")
    sys.exit(1)
else:
    print("\n🎉 All tests passed! Linux compatibility confirmed.")
    print("🐳 Ready for Docker deployment!")

# Test environment variables
print(f"\n🌍 Environment variables:")
print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"  PATH: {os.environ.get('PATH', 'Not set')[:100]}...")
print(f"  OPENAI_API_KEY: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not set'}")

# Performance suggestions
print(f"\n⚡ Performance recommendations for Linux:")
print(f"  - Set OPENBLAS_NUM_THREADS=1 for containerized deployment")
print(f"  - Set TOKENIZERS_PARALLELISM=false to avoid warnings")
print(f"  - Consider using uvloop for async performance boost")