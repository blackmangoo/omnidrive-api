import io
import os
import sys
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

# Ensure api folder is on sys.path
api_dir = str(Path(__file__).resolve().parent.parent)
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from main import app

client = TestClient(app)


def test_health_check_get():
    """Verify /health returns 200 OK and expected status keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"
    assert "model_loaded" in data
    assert "model_available" in data


def test_health_check_post():
    """Verify POST /health works identically for clients using POST probes."""
    response = client.post("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"


def test_predict_requires_file():
    """Verify /predict returns 422 when no file is supplied."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_rejects_non_image():
    """Verify /predict returns 400 when an invalid file type is uploaded."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "File is not a recognised image" in response.json().get("detail", "")


def test_predict_rejects_corrupted_image():
    """Verify /predict returns 400 Bad Request on corrupted image bytes."""
    files = {"file": ("damaged.jpg", b"GIF89a corrupt bytes", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_valid_synthetic_image():
    """Verify /predict correctly decodes a valid image and runs inference or returns 503 if model weights missing."""
    img_byte_arr = io.BytesIO()
    test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    test_img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    files = {"file": ("synthetic_part.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    if response.status_code == 200:
        data = response.json()
        assert data.get("success") is True
        assert "top_prediction" in data
        assert "top_confidence" in data
        assert isinstance(data.get("all_predictions"), list)
        assert "inference_time_ms" in data
    elif response.status_code == 503:
        # Acceptable in test environments if .pt weights are not bundled
        assert "not available" in response.json().get("detail", "")
    else:
        pytest.fail(f"Unexpected status code {response.status_code}: {response.text}")


def test_ingest_data_fail_closed_unauthorized():
    """Verify /ingest_data rejects unauthenticated requests with HTTP 403."""
    response = client.get("/ingest_data")
    assert response.status_code == 403

    response_bad_token = client.post("/ingest_data?token=invalid_token_123")
    assert response_bad_token.status_code == 403


def test_chat_requires_body():
    """Verify /chat requires a JSON body containing query."""
    response = client.post("/chat", json={})
    assert response.status_code == 422
