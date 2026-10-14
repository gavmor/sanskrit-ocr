"""
Smoke tests for Sanskrit OCR model inference and decoding.
"""

import base64
from pathlib import Path
import pytest
import requests

from devanagari_validator import decode_output

SAMPLE_IMG = Path("get_predictions/samples/23.jpg")
SERVER_URL = "http://127.0.0.1:8501/v1/models/sanskrit:predict"


def test_sample_image_exists():
    assert SAMPLE_IMG.exists()
    assert SAMPLE_IMG.stat().st_size > 1000


def test_live_server_prediction():
    try:
        r = requests.get("http://127.0.0.1:8501/v1/models/sanskrit", timeout=1.0)
        server_ready = (r.status_code == 200)
    except Exception:
        server_ready = False

    if not server_ready:
        pytest.skip("TensorFlow Serving daemon is not running on 127.0.0.1:8501")

    with open(SAMPLE_IMG, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode("utf-8")

    resp = requests.post(
        SERVER_URL,
        json={"signature_name": "serving_default", "inputs": [{"b64": b64}]},
        timeout=10.0,
    )
    assert resp.status_code == 200
    res = resp.json()
    assert "outputs" in res
    assert "output" in res["outputs"]

    decoded = decode_output(res["outputs"]["output"])
    assert decoded == "धर्माब्धिस्थमुदाहरणम्"
