#!/usr/bin/env python3
"""
Entry point for the AI-Powered Legal Search Engine.

This script launches the Streamlit application.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
from src.app import main

if __name__ == "__main__":
    main()