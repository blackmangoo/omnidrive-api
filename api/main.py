from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
import os
import time
from PIL import Image

app = FastAPI(title="Car Parts Classification API", version="1.0.0")

# Enable CORS for mobile app/frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained custom YOLO11 Large model exactly once at startup
try:
    print("Loading Custom YOLO11 Large Model into memory...")
    # Update this path if the script is run from a different directory
    model = YOLO("models/car_parts_large_v1.pt")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Car Parts Classification API is Running",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_car_part(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the top 3 predicted car parts
    along with their confidence scores.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Validate file type — accept explicit image/* MIME types
    # OR application/octet-stream (what Android camera package sends)
    # OR fall back to a known image file extension.
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()

    is_image_mime = file.content_type and file.content_type.startswith("image/")
    is_octet = file.content_type in (None, "", "application/octet-stream")
    is_known_ext = ext in ALLOWED_EXTENSIONS

    if not (is_image_mime or (is_octet and is_known_ext)):
        raise HTTPException(
            status_code=400,
            detail=f"File is not a recognised image (content_type={file.content_type!r}, ext={ext!r})."
        )

    try:
        # Read the image bytes directly into a PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Run YOLO inference
        start_time = time.time()
        results = model.predict(source=image, imgsz=224, verbose=False)
        inference_time = (time.time() - start_time) * 1000 # in ms
        
        # Process the single result
        result = results[0]
        
        # Get the top 5 predicted classes and their probabilities
        top5_indices = result.probs.top5
        top5_confs = result.probs.top5conf.tolist()
        
        predictions = []
        for idx, conf in zip(top5_indices, top5_confs):
            class_name = result.names[idx]
            predictions.append({
                "class": class_name,
                "confidence": round(float(conf) * 100, 2) # Return as percentage
            })
            
        return JSONResponse(content={
            "success": True,
            "top_prediction": predictions[0]["class"],
            "top_confidence": predictions[0]["confidence"],
            "all_predictions": predictions,
            "inference_time_ms": round(inference_time, 2)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# This allows running the file directly with Python
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
