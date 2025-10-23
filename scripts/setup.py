#!/usr/bin/env python3
"""
Setup script for the AI-Powered Legal Search Engine.

This script initializes the application environment and validates setup.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.config import get_config
    from services.database import get_database_service
    from services.vector_store import get_vector_store_service
    from services.embedding import get_embedding_service
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

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
        'faiss-cpu',
        'cohere',
        'groq',
        'PyMuPDF',
        'python-dotenv',
        'pydantic',
        'numpy',
        'pandas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
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
        print("⚠️  .env file not found. Creating from template...")

        # Copy from example if exists
        example_file = Path(".env.example")
        if example_file.exists():
            import shutil
            shutil.copy2(example_file, env_file)
            print("✅ Created .env from .env.example")
        else:
            # Create basic .env file
            with open(env_file, 'w') as f:
                f.write("""# API Keys (required)
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
DATABASE_PATH=src/data/.legal_db.sqlite
FAISS_INDEX_PATH=src/data/.faiss_index
""")
            print("✅ Created basic .env file")

        print("⚠️  Please edit .env file and add your API keys")
        return False

    # Check if API keys are set
    try:
        config = get_config()
        if not config.cohere_api_key or config.cohere_api_key == "your_cohere_api_key_here":
            print("❌ COHERE_API_KEY not set in .env")
            return False

        if not config.groq_api_key or config.groq_api_key == "your_groq_api_key_here":
            print("❌ GROQ_API_KEY not set in .env")
            return False

        print("✅ Environment variables configured")
        return True

    except Exception as e:
        print(f"❌ Error checking environment variables: {e}")
        return False

def initialize_database():
    """Initialize the database."""
    try:
        db_service = get_database_service()
        success = db_service.initialize_database()

        if success:
            print("✅ Database initialized successfully")
            return True
        else:
            print("❌ Failed to initialize database")
            return False

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def initialize_vector_store():
    """Initialize the vector store."""
    try:
        vector_service = get_vector_store_service()
        success = vector_service.initialize_index()

        if success:
            stats = vector_service.get_index_stats()
            print(f"✅ Vector store initialized ({stats.total_vectors} vectors)")
            return True
        else:
            print("❌ Failed to initialize vector store")
            return False

    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        return False

def validate_api_keys():
    """Validate API keys."""
    try:
        # Test Cohere API
        embedding_service = get_embedding_service()
        if embedding_service.validate_api_key():
            print("✅ Cohere API key valid")
            cohere_valid = True
        else:
            print("❌ Cohere API key invalid")
            cohere_valid = False

        # Groq API validation would require actual API call
        # For now, just check if key is present
        config = get_config()
        if config.groq_api_key:
            print("✅ Groq API key present")
            groq_valid = True
        else:
            print("❌ Groq API key missing")
            groq_valid = False

        return cohere_valid and groq_valid

    except Exception as e:
        print(f"❌ API key validation failed: {e}")
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
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 AI-Powered Legal Search Engine Setup")
    print("=" * 50)

    setup_logging()

    # Check Python version
    print("\n📋 Checking Python version...")
    if not check_python_version():
        sys.exit(1)

    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n💡 Install missing dependencies with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Create directories
    print("\n📁 Creating directories...")
    create_directories()

    # Check environment variables
    print("\n🔧 Checking environment configuration...")
    env_ok = check_environment_variables()

    if not env_ok:
        print("\n💡 Please configure your API keys in the .env file")
        print("   1. Get API keys from cohere.com and groq.com")
        print("   2. Edit .env file with your keys")
        print("   3. Run setup.py again")
        sys.exit(1)

    # Initialize database
    print("\n🗄️ Initializing database...")
    if not initialize_database():
        sys.exit(1)

    # Initialize vector store
    print("\n🔍 Initializing vector store...")
    if not initialize_vector_store():
        sys.exit(1)

    # Validate API keys
    print("\n🔑 Validating API keys...")
    if not validate_api_keys():
        print("\n💡 Please check your API keys in the .env file")
        sys.exit(1)

    # Success
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n🚀 Start the application with:")
    print("   streamlit run app.py")
    print("\n🌐 The application will open in your web browser at:")
    print("   http://localhost:8501")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()