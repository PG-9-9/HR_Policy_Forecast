#!/usr/bin/env python3
"""
Smart dependency checker and auto-installer for policy_api environment
Checks for missing packages and installs them automatically
"""

import subprocess
import sys
import importlib
import pkg_resources

# Required packages for chatbot functionality (extracted from working environment)
REQUIRED_PACKAGES = {
    'fastapi': '0.116.2',
    'uvicorn': '0.35.0', 
    'python-multipart': '0.0.20',
    'openai': '1.108.0',
    'jinja2': '3.1.6',
    'python-dotenv': '1.1.1',
    'pandas': '2.3.2',
    'numpy': '2.1.3',
    'requests': '2.32.3',
    'sentence-transformers': '5.1.0',
    'faiss-cpu': '1.12.0',
    'rank-bm25': '0.2.2',
    'beautifulsoup4': '4.12.3',
    'pdfminer.six': 'latest'  # Version detection not working, but module available
}

# Import name mappings (package name vs import name)
IMPORT_MAPPINGS = {
    'python-multipart': 'multipart',
    'beautifulsoup4': 'bs4',
    'pdfminer.six': 'pdfminer',
    'faiss-cpu': 'faiss',
    'python-dotenv': 'dotenv'
}

def check_package_installed(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = IMPORT_MAPPINGS.get(package_name, package_name.replace('-', '_'))
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def get_installed_version(package_name):
    """Get the installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def install_package(package_name, version):
    """Install a package using pip"""
    package_spec = f"{package_name}=={version}"
    print(f"📦 Installing {package_spec}...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_spec
        ], capture_output=True, text=True, check=True)
        print(f"✅ Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_spec}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔍 Checking policy_api environment dependencies...")
    print(f"🐍 Python: {sys.executable}")
    print()
    
    missing_packages = []
    version_mismatches = []
    
    # Check each required package
    for package_name, required_version in REQUIRED_PACKAGES.items():
        import_name = IMPORT_MAPPINGS.get(package_name, package_name.replace('-', '_'))
        
        if check_package_installed(package_name, import_name):
            installed_version = get_installed_version(package_name)
            if installed_version:
                if installed_version == required_version:
                    print(f"✅ {package_name}: {installed_version}")
                else:
                    print(f"⚠️ {package_name}: {installed_version} (expected {required_version})")
                    version_mismatches.append((package_name, installed_version, required_version))
            else:
                print(f"✅ {package_name}: Available (version unknown)")
        else:
            print(f"❌ {package_name}: Not installed")
            missing_packages.append((package_name, required_version))
    
    print()
    
    # Install missing packages
    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        for package_name, version in missing_packages:
            if not install_package(package_name, version):
                print(f"❌ Failed to install {package_name}")
                return False
        print()
    
    # Handle version mismatches
    if version_mismatches:
        print("⚠️ Version mismatches detected:")
        for package_name, installed, required in version_mismatches:
            print(f"  {package_name}: {installed} → {required}")
        
        response = input("Update packages to required versions? (y/n): ")
        if response.lower() == 'y':
            for package_name, _, required_version in version_mismatches:
                install_package(package_name, required_version)
    
    # Final validation
    print("\n🧪 Final validation...")
    all_good = True
    
    # Test core imports
    test_imports = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("multipart", "Multipart form data"),
        ("openai", "OpenAI API client"),
        ("jinja2", "Template engine"),
        ("dotenv", "Environment variables"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("requests", "HTTP client"),
        ("sentence_transformers", "Sentence embeddings"),
        ("faiss", "Vector search"),
        ("rank_bm25", "BM25 ranking"),
        ("bs4", "HTML parsing"),
        ("pdfminer", "PDF parsing")
    ]
    
    for module_name, description in test_imports:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: {description}")
        except ImportError as e:
            print(f"❌ {module_name}: Failed to import - {e}")
            all_good = False
    
    # Test app import
    print("\n🔧 Testing app import...")
    try:
        from app.main import app
        print("✅ App imports successfully")
    except ImportError as e:
        print(f"❌ App import failed: {e}")
        all_good = False
    
    if all_good:
        print("\n🎉 All dependencies are satisfied!")
        print("🚀 Ready to start the chatbot server!")
        print("\nTo start the server:")
        print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return True
    else:
        print("\n❌ Some dependencies are still missing or broken")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)