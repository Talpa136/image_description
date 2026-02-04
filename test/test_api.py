import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main as m  # main.py è in root

client = TestClient(m.app)


def make_png_bytes(size=(10, 20), color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_reject_non_image_type_415():
    r = client.post("/images/describe", files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_empty_file_400():
    r = client.post("/images/describe", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_too_large_413():
    big = b"x" * (10 * 1024 * 1024 + 1)
    r = client.post("/images/describe", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


@pytest.mark.parametrize("mode", ["local", "remote"])
def test_ok_image_with_mocked_llm(monkeypatch, mode):
    # Forziamo la modalità via env (il codice legge LLM_MODE a runtime)
    os.environ["LLM_MODE"] = mode

    async def fake_generate_local(prompt: str):
        return "Immagine 10x20 px, colori dominanti: rosso, rosso, rosso.", "fake-local-model"

    async def fake_generate_remote(prompt: str):
        return "Immagine 10x20 px, colori dominanti: rosso, rosso, rosso.", "fake-remote-model"

    # patch delle funzioni nel modulo main
    monkeypatch.setattr(m, "generate_local", fake_generate_local)
    monkeypatch.setattr(m, "generate_remote", fake_generate_remote)

    img_bytes = make_png_bytes()
    r = client.post("/images/describe", files={"file": ("img.png", img_bytes, "image/png")})
    assert r.status_code == 200

    data = r.json()
    assert data["mode"] == mode
    assert "description" in data
    assert isinstance(data["description"], str)

    if mode == "local":
        assert data["model"] == "fake-local-model"
    else:
        assert data["model"] == "fake-remote-model"