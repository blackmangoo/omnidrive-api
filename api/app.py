"""
OmniDrive API entrypoint.
Re-exports the canonical FastAPI application from main.py for compatibility
with Docker containers and deployment targets expecting `app:app`.
"""
import os
import uvicorn
from main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
