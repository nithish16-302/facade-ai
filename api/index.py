import sys
import os

# Make sure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app
