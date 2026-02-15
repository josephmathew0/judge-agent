import pytest


fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from judge_agent.web import app


def test_web_text_upload_runs():
    client = TestClient(app)

    files = {"file": ("sample.txt", b"Here are 3 focus tips. Like and subscribe!")}
    data = {
        "content_type": "text",
        "fps_sample": "1.0",
        "max_frames": "60",
        "debug": "false",
    }

    resp = client.post("/judge", files=files, data=data)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["origin_prediction"]["label"] in {"ai_generated", "human_generated"}
    assert isinstance(payload["virality_score"], int)
    assert "distribution_analysis" in payload


def test_web_rejects_invalid_fps_sample():
    client = TestClient(app)

    files = {"file": ("sample.txt", b"test")}
    data = {
        "content_type": "video",
        "fps_sample": "0",
        "max_frames": "60",
        "debug": "false",
    }

    resp = client.post("/judge", files=files, data=data)
    assert resp.status_code == 400
    assert "fps_sample must be greater than 0" in resp.json()["error"]
