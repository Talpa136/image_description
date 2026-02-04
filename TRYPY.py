from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Literal

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ----------------------------
# Config
# ----------------------------
MAX_BYTES = 10 * 1024 * 1024  # 10MB
N_COLORS = 3
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}  # pdf solo per test 415/validazioni

def get_mode() -> str:
    return os.getenv("LLM_MODE", "local").lower()

# Local (Ollama)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# Remote (OpenAI-compatible Chat Completions)
REMOTE_BASE_URL = os.getenv("REMOTE_BASE_URL", "https://api.openai.com/v1")
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "gpt-4o-mini")
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY")  # richiesto se MODE=remote

app = FastAPI()


# ----------------------------
# Middleware: early size check
# ----------------------------
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BYTES:
            return JSONResponse(status_code=413, content={"detail": "File too large. Max 10MB."})
        return await call_next(request)


app.add_middleware(LimitUploadSizeMiddleware)


# ----------------------------
# Utilities: colors + image validation
# ----------------------------
def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PALETTE_IT = [
    ("nero", "#000000"),
    ("bianco", "#ffffff"),
    ("grigio", "#808080"),
    ("beige", "#f5f5dc"),
    ("marrone", "#8b4513"),
    ("ocra", "#cc7722"),
    ("giallo", "#ffff00"),
    ("arancione", "#ffa500"),
    ("rosso", "#ff0000"),
    ("verde", "#008000"),
    ("blu", "#0000ff"),
    ("viola", "#800080"),
    ("rosa", "#ffc0cb"),
]


def closest_color_name_it(hex_str: str) -> str:
    r, g, b = hex_to_rgb(hex_str)

    def dist2(p_hex: str) -> int:
        rr, gg, bb = hex_to_rgb(p_hex)
        return (r - rr) ** 2 + (g - gg) ** 2 + (b - bb) ** 2

    name, _ = min(PALETTE_IT, key=lambda item: dist2(item[1]))
    return name


def extract_metadata_with_pillow(image_bytes: bytes, n_colors: int = 3):
    # Validazione: è un'immagine?
    try:
        with Image.open(BytesIO(image_bytes)) as im_check:
            im_check.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid file: not a recognized image format")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file: corrupted or unreadable image")

    # Riapriamo per elaborare
    with Image.open(BytesIO(image_bytes)) as im:
        im_rgb = im.convert("RGB")
        width, height = im_rgb.size

        small = im_rgb.copy()
        small.thumbnail((200, 200))

        quant = small.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        palette = quant.getpalette()
        color_counts = quant.getcolors()

        dominant = []
        if color_counts:
            color_counts.sort(key=lambda x: x[0], reverse=True)
            for count, idx in color_counts[:n_colors]:
                r = palette[3 * idx]
                g = palette[3 * idx + 1]
                b = palette[3 * idx + 2]
                dominant.append({"hex": rgb_to_hex((r, g, b)), "weight": count})

        return width, height, dominant


def build_prompt(width: int, height: int, dominant_colors: list[dict]) -> str:
    if dominant_colors:
        color_names = ", ".join(closest_color_name_it(c["hex"]) for c in dominant_colors)
    else:
        color_names = "N/A"

    return (
        "Sei un convertitore di metadati in una frase.\n"
        "Usa SOLO i dati forniti. NON descrivere contenuti dell'immagine.\n"
        "Formato obbligatorio (una sola frase, in italiano):\n"
        "\"Immagine {WIDTH}x{HEIGHT} px, colori dominanti: {COLORS}.\"\n"
        f"Dati: WIDTH={width}, HEIGHT={height}, COLORS={color_names}\n"
        "Rispondi SOLO con la frase nel formato indicato, senza altro testo."
    )


# ----------------------------
# LLM Clients (local / remote)
# ----------------------------
async def generate_local(prompt: str) -> tuple[str, str]:
    """Returns (description, model_used)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 60},
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip(), OLLAMA_MODEL
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Cannot reach local model (Ollama). Is it running?")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Local model error: {e.response.text}")


async def generate_remote(prompt: str) -> tuple[str, str]:
    """OpenAI-compatible Chat Completions. Returns (description, model_used)."""
    if not REMOTE_API_KEY:
        raise HTTPException(status_code=500, detail="REMOTE_API_KEY is not set (required for LLM_MODE=remote)")

    url = f"{REMOTE_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {REMOTE_API_KEY}"}
    payload = {
        "model": REMOTE_MODEL,
        "messages": [
            {"role": "system", "content": "Sei un assistente che converte metadati in una sola frase."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 80,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Cannot reach remote LLM provider")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Remote LLM error: {e.response.text}")

    try:
        text = data["choices"][0]["message"]["content"].strip()
        return text, REMOTE_MODEL
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected remote LLM response: {data}")


async def generate_description(prompt: str) -> tuple[str, str, str]:
    mode = get_mode()
    if mode == "local":
        desc, model = await generate_local(prompt)
        return desc, "local", model
    if mode == "remote":
        desc, model = await generate_remote(prompt)
        return desc, "remote", model
    raise HTTPException(status_code=500, detail=f"Invalid LLM_MODE: {mode}")

# ----------------------------
# API
# ----------------------------
class DescribeResponse(BaseModel):
    description: str
    mode: Literal["local", "remote"]
    model: str


@app.post("/images/describe", response_model=DescribeResponse)
async def upload_image_and_describe(file: UploadFile = File(...)):
    # Check MIME dichiarato dal client (opzionale ma utile)
    if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported media type (expected jpeg/png/webp)")

    # Lettura con limite 10MB
    size = 0
    buf = bytearray()
    chunk_size = 1024 * 1024  # 1MB

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Max 10MB.")
        buf.extend(chunk)

    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validazione + metadati via Pillow
    width, height, dominant_colors = extract_metadata_with_pillow(bytes(buf), N_COLORS)

    # Prompt -> LLM (local or remote)
    prompt = build_prompt(width, height, dominant_colors)
    description, mode_used, model_used = await generate_description(prompt)

    return DescribeResponse(description=description, mode=mode_used, model=model_used)
