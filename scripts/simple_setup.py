#!/usr/bin/env python3
"""
Simple setup script for the AI-Powered Legal Search Engine.

This script initializes the application environment and validates setup.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("[ERROR] Python 3.11+ is required")
        return False

    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'faiss_cpu',
        'cohere',
        'groq',
        'PyMuPDF',
        'dotenv',
        'pydantic',
        'numpy',
        'pandas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PyMuPDF':
                import fitz
            elif package == 'dotenv':
                import dotenv
            elif package == 'faiss_cpu':
                import faiss
            else:
                __import__(package.lower().replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True

def check_environment_variables():
    """Check required environment variables."""
    env_file = Path(".env")

    if not env_file.exists():
        print("[WARNING] .env file not found. Creating from template...")

        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("""# API Keys (required)
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
DATABASE_PATH=src/data/.legal_db.sqlite
FAISS_INDEX_PATH=src/data/.faiss_index
""")
        print("[OK] Created basic .env file")
        print("[WARNING] Please edit .env file and add your API keys")
        return False

    # Check if API keys are set
    try:
        from dotenv import load_dotenv
        load_dotenv()

        cohere_key = os.getenv('COHERE_API_KEY')
        groq_key = os.getenv('GROQ_API_KEY')

        if not cohere_key or cohere_key == "your_cohere_api_key_here":
            print("[ERROR] COHERE_API_KEY not set in .env")
            return False

        if not groq_key or groq_key == "your_groq_api_key_here":
            print("[ERROR] GROQ_API_KEY not set in .env")
            return False

        print("[OK] Environment variables configured")
        return True

    except Exception as e:
        print(f"[ERROR] Error checking environment variables: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "src/data",
        "src/data/uploads",
        "logs",
        "tests/fixtures/sample_legal_docs"
    ]

    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {directory}")

def main():
    """Main setup function."""
    print("AI-Powered Legal Search Engine Setup")
    print("=" * 50)

    setup_logging()

    # Check Python version
    print("\n[STEP 1] Checking Python version...")
    if not check_python_version():
        sys.exit(1)

    # Check dependencies
    print("\n[STEP 2] Checking dependencies...")
    if not check_dependencies():
        print("\nInstall missing dependencies with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Create directories
    print("\n[STEP 3] Creating directories...")
    create_directories()

    # Check environment variables
    print("\n[STEP 4] Checking environment configuration...")
    env_ok = check_environment_variables()

    if not env_ok:
        print("\nPlease configure your API keys in the .env file")
        print("1. Get API keys from cohere.com and groq.com")
        print("2. Edit .env file with your keys")
        print("3. Run setup.py again")
        sys.exit(1)

    # Success
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nStart the application with:")
    print("   streamlit run app.py")
    print("\nThe application will open in your web browser at:")
    print("   http://localhost:8501")

if __name__ == "__main__":
    main()