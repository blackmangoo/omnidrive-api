from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
import io
import os
import time
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="OmniDrive API")

# Enable CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load YOLO model at startup (CPU) ──
try:
    print("Loading Custom YOLO11 Large Model into memory (CPU)...")
    yolo_model = YOLO("car_parts_large_v1.pt")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    try:
        yolo_model = YOLO("models/car_parts_large_v1.pt")
        print("Model loaded successfully from models/ dir!")
    except:
        yolo_model = None

@app.post("/health")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "message": "Car Parts Classification API is Running",
        "model_loaded": yolo_model is not None
    }

@app.post("/predict")
def predict_car_part(file: UploadFile = File(...)):
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

        start_time = time.time()
        results = yolo_model.predict(source=image, imgsz=224, verbose=False)
        inference_time = (time.time() - start_time) * 1000

        result = results[0]
        predictions = [
            {"class": result.names[idx], "confidence": round(float(conf) * 100, 2)}
            for idx, conf in zip(result.probs.top5, result.probs.top5conf.tolist())
        ]

        return JSONResponse(content={
            "success": True,
            "top_prediction": predictions[0]["class"],
            "top_confidence": predictions[0]["confidence"],
            "all_predictions": predictions,
            "inference_time_ms": round(inference_time, 2)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── RAG Chatbot Endpoint ──
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://cqeubytgsrxdkfejxvan.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_KEY else None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    if not supabase_client or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys")

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
        res = requests.post(url, json={"model": "models/gemini-embedding-2", "content": {"parts": [{"text": request.query}]}, "outputDimensionality": 768}).json()
        query_embedding = res["embedding"]["values"]

        response = supabase_client.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 3}).execute()
        docs = response.data
        context_text = "\n\n".join([doc['content'] for doc in docs]) if docs else "No specific DIY documentation found."

        prompt = f"You are a helpful AI mechanic. Answer based on this docs:\n{context_text}\n\nQuestion: {request.query}"
        chat_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
        chat_res = requests.post(chat_url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        
        return {"success": True, "answer": chat_res["candidates"][0]["content"]["parts"][0]["text"], "retrieved_docs": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest_data")
def trigger_ingestion():
    import subprocess
    result = subprocess.run(["python", "rag_ingest.py"], capture_output=True, text=True)
    return {"success": result.returncode == 0, "logs": result.stdout if result.returncode == 0 else result.stderr}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
