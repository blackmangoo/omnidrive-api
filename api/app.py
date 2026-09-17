"""
OmniDrive API entrypoint.
Re-exports the canonical FastAPI application from main.py for compatibility
with Docker containers, local execution, and deployment targets expecting `app:app`.
"""
import os
import sys

# Ensure the directory containing this file is in sys.path (Finding #13)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from main import app
except ImportError:
    from api.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)
