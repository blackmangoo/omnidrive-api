import os
# Bound memory arenas and thread-pools before loading native libs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import gc
import io
import time
from PIL import Image
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

# Load environment variables
load_dotenv()

app = FastAPI(title="OmniDrive Car Parts Classification & RAG API", version="1.0.0")

# Enable CORS for mobile app/frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy Loaded YOLO Model ───────────────────────────────────────────────────
_yolo_model = None

def get_yolo_model():
    global _yolo_model
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
    return {
        "status": "online",
        "message": "Car Parts Classification API is Running",
        "model_loaded": _yolo_model is not None,
        "model_available": model_on_disk,
    }

@app.post("/predict")
async def predict_car_part(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the top 5 predicted car parts
    along with their confidence scores.
    """
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

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

        start_time = time.time()
        with torch.inference_mode():
            results = model.predict(source=image, imgsz=224, verbose=False)
        inference_time = (time.time() - start_time) * 1000

        result = results[0]
        top5_indices = result.probs.top5
        top5_confs = result.probs.top5conf.tolist()

        predictions = []
        for idx, conf in zip(top5_indices, top5_confs):
            class_name = result.names[idx]
            predictions.append({
                "class": class_name,
                "confidence": round(float(conf) * 100, 2),
            })

        # Prompt Python to reclaim temporary image tensor buffers
        del contents
        del image
        gc.collect()

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
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ── RAG Chatbot Endpoint ──
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://cqeubytgsrxdkfejxvan.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_KEY else None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    if not supabase_client or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys (SUPABASE_KEY or GEMINI_API_KEY)")

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
        res = requests.post(
            url,
            json={
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": request.query}]},
                "outputDimensionality": 768,
            },
            timeout=15,
        ).json()

        if "error" in res:
            raise HTTPException(status_code=502, detail=f"Embedding API error: {res['error'].get('message')}")

        query_embedding = res["embedding"]["values"]

        response = supabase_client.rpc(
            "match_documents",
            {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 3},
        ).execute()
        docs = response.data
        context_text = "\n\n".join([doc["content"] for doc in docs]) if docs else "No specific DIY documentation found."

        prompt = (
            "You are OmniDrive's expert AI Master Mechanic. Your role is to provide accurate, "
            "step-by-step automotive guidance, diagnostics, and maintenance advice.\n\n"
            "Reference the verified technical documentation below to answer the inquiry. "
            "If the documentation does not directly answer the inquiry, provide helpful automotive "
            "best practices and emphasize workshop safety.\n\n"
            f"<technical_documentation>\n{context_text}\n</technical_documentation>\n\n"
            f"<user_question>\n{request.query}\n</user_question>"
        )
        chat_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
        chat_res = requests.post(
            chat_url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
        ).json()

        if "error" in chat_res:
            error_msg = chat_res["error"].get("message", "Gemini API error")
            raise HTTPException(status_code=502, detail=f"AI model error: {error_msg}")

        candidates = chat_res.get("candidates", [])
        if not candidates or "content" not in candidates[0]:
            return {
                "success": True,
                "answer": "I could not generate a response for that inquiry. Please try rephrasing.",
                "retrieved_docs": len(docs),
            }

        answer_text = candidates[0]["content"]["parts"][0]["text"]
        return {"success": True, "answer": answer_text, "retrieved_docs": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest_data")
@app.post("/ingest_data")
def trigger_ingestion(token: str = ""):
    expected = os.environ.get("INGEST_SECRET", "")
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid ingestion token")
    import subprocess
    result = subprocess.run(["python", "rag_ingest.py"], capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "logs": result.stdout if result.returncode == 0 else result.stderr,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
