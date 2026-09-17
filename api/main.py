import os
# Bound memory arenas and thread-pools before loading native libs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import asyncio
import io
from pathlib import Path
import subprocess
import sys
import threading
import time
from PIL import Image, UnidentifiedImageError
import requests
import torch
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# Enforce single-threaded PyTorch CPU runtime to stay well within 512MB RAM
torch.set_num_threads(1)

# Finding #7: Load environment variables with path anchored to this file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="OmniDrive Car Parts Classification & RAG API", version="1.0.0")

# Enable CORS for mobile app/frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Thread-Safe Lazy Loaded YOLO Model (Finding #5) ───────────────────────────
_yolo_model = None
_model_lock = threading.Lock()

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        with _model_lock:
            if _yolo_model is None:
                candidate_paths = [
                    os.path.join(os.path.dirname(__file__), "models", "car_parts_large_v1.pt"),
                    os.path.join(os.path.dirname(__file__), "car_parts_large_v1.pt"),
                    "models/car_parts_large_v1.pt",
                    "car_parts_large_v1.pt",
                ]
                chosen_path = None
                for p in candidate_paths:
                    if os.path.exists(p):
                        chosen_path = p
                        break

                if chosen_path:
                    print(f"Loading YOLO11 model into CPU memory from {chosen_path}...")
                    _yolo_model = YOLO(chosen_path)
                    print("YOLO11 model loaded successfully!")
                else:
                    print("Warning: car_parts_large_v1.pt model file not found on disk.")
    return _yolo_model

@app.get("/")
@app.post("/health")
@app.get("/health")
def health_check():
    model_on_disk = any(
        os.path.exists(p)
        for p in [
            os.path.join(os.path.dirname(__file__), "models", "car_parts_large_v1.pt"),
            os.path.join(os.path.dirname(__file__), "car_parts_large_v1.pt"),
            "models/car_parts_large_v1.pt",
            "car_parts_large_v1.pt",
        ]
    )
    # Finding #2: Return model_loaded as True if model is already loaded OR ready on disk
    # so mobile apps querying /health recognize the API as Online immediately.
    is_ready = (_yolo_model is not None) or model_on_disk
    return {
        "status": "online",
        "message": "Car Parts Classification API is Running",
        "model_loaded": is_ready,
        "model_available": model_on_disk,
    }

# Finding #6: 10 MB upload cap to prevent memory exhaustion
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

def _sync_yolo_inference(model: YOLO, image: Image.Image):
    """CPU-bound inference executed off the asyncio event loop."""
    start_time = time.time()
    with torch.inference_mode():
        results = model.predict(source=image, imgsz=224, verbose=False)
    inference_time = (time.time() - start_time) * 1000

    result = results[0]
    top5_indices = result.probs.top5
    top5_confs = result.probs.top5conf.tolist()

    predictions = [
        {
            "class": result.names[idx],
            "confidence": round(float(conf) * 100, 2),
        }
        for idx, conf in zip(top5_indices, top5_confs)
    ]
    return predictions, inference_time

@app.post("/predict")
async def predict_car_part(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the top 5 predicted car parts
    along with their confidence scores.
    """
    # Finding #5: Retrieve model safely
    model = get_yolo_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model file (car_parts_large_v1.pt) is not available on server.",
        )

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()

    is_image_mime = file.content_type and file.content_type.startswith("image/")
    is_octet = file.content_type in (None, "", "application/octet-stream")
    is_known_ext = ext in ALLOWED_EXTENSIONS

    if not (is_image_mime or (is_octet and is_known_ext)):
        raise HTTPException(
            status_code=400,
            detail=f"File is not a recognised image (content_type={file.content_type!r}, ext={ext!r}).",
        )

    # Finding #6: Bounded read to avoid OOM
    contents = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum size limit of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    # Finding #8: Catch UnidentifiedImageError and return 400 Bad Request
    try:
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is corrupted or not a valid image format.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading image: {str(e)}",
        )

    try:
        # Finding #4: Run CPU-bound inference in a threadpool worker, not on asyncio loop
        predictions, inference_time = await asyncio.to_thread(_sync_yolo_inference, model, image)

        # Finding #11: Omit synchronous stop-the-world gc.collect() to eliminate latency spikes
        del contents
        del image

        return JSONResponse(
            content={
                "success": True,
                "top_prediction": predictions[0]["class"],
                "top_confidence": predictions[0]["confidence"],
                "all_predictions": predictions,
                "inference_time_ms": round(inference_time, 2),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")

# ── RAG Chatbot Endpoint ──
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://cqeubytgsrxdkfejxvan.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_KEY else None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    query: str

def _sync_chat_with_rag(query: str):
    """Synchronous network I/O and vector search executed off the asyncio event loop."""
    # 1. Embedding request with status and error validation (Finding #9)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
    try:
        res = requests.post(
            url,
            json={
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": query}]},
                "outputDimensionality": 768,
            },
            timeout=15,
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Gemini embedding request timed out.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach Gemini embedding endpoint: {str(e)}")

    if res.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embedding API error (HTTP {res.status_code}): {res.text[:200]}",
        )

    try:
        embedding_data = res.json()
        query_embedding = embedding_data["embedding"]["values"]
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON response from Gemini embedding API.")

    # 2. Supabase pgvector search
    try:
        response = supabase_client.rpc(
            "match_documents",
            {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 3},
        ).execute()
        docs = response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase vector search error: {str(e)}")

    context_text = "\n\n".join([doc["content"] for doc in docs]) if docs else "No specific DIY documentation found."

    prompt = (
        "You are OmniDrive's expert AI Master Mechanic. Your role is to provide accurate, "
        "step-by-step automotive guidance, diagnostics, and maintenance advice.\n\n"
        "Reference the verified technical documentation below to answer the inquiry. "
        "If the documentation does not directly answer the inquiry, provide helpful automotive "
        "best practices and emphasize workshop safety.\n\n"
        f"<technical_documentation>\n{context_text}\n</technical_documentation>\n\n"
        f"<user_question>\n{query}\n</user_question>"
    )

    # 3. Gemini Chat Completion with status and error validation (Finding #9)
    chat_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    try:
        chat_res = requests.post(
            chat_url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Gemini text generation request timed out.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach Gemini generation endpoint: {str(e)}")

    if chat_res.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini generation API error (HTTP {chat_res.status_code}): {chat_res.text[:200]}",
        )

    try:
        chat_data = chat_res.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON received from Gemini generation API.")

    if "error" in chat_data:
        error_msg = chat_data["error"].get("message", "Gemini API error")
        raise HTTPException(status_code=502, detail=f"AI model error: {error_msg}")

    candidates = chat_data.get("candidates", [])
    if not candidates or "content" not in candidates[0]:
        return {
            "success": True,
            "answer": "I could not generate a response for that inquiry. Please try rephrasing.",
            "retrieved_docs": len(docs),
        }

    answer_text = candidates[0]["content"]["parts"][0]["text"]
    return {"success": True, "answer": answer_text, "retrieved_docs": len(docs)}

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    if not supabase_client or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys (SUPABASE_KEY or GEMINI_API_KEY)")

    # Finding #3: Run synchronous requests/DB network operations in threadpool so asyncio loop is unblocked
    return await asyncio.to_thread(_sync_chat_with_rag, request.query)

# Finding #1: Fail-closed authentication (reject missing or mismatched secret)
@app.get("/ingest_data")
@app.post("/ingest_data")
def trigger_ingestion(token: str = ""):
    expected = os.environ.get("INGEST_SECRET", "")
    if not expected or token != expected:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Ingestion requires a valid INGEST_SECRET configured in environment.",
        )

    # Finding #10: Use sys.executable, specify cwd, enforce timeout, and avoid leaking traces
    try:
        result = subprocess.run(
            [sys.executable, "rag_ingest.py"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "success": result.returncode == 0,
            "message": "Ingestion completed successfully." if result.returncode == 0 else "Ingestion process encountered an error.",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Ingestion script timed out after 300 seconds.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute ingestion: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
