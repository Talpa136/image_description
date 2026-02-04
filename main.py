from __future__ import annotations

import os
from io import BytesIO
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ------------------------------------------------------------
# Load env from a selectable file:
#   ENV_FILE=.env.local  OR  ENV_FILE=.env.remote
# Default: .env (if present)
# ------------------------------------------------------------
ENV_FILE = os.getenv("ENV_FILE", ".env")
load_dotenv(ENV_FILE, override=True)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
MAX_BYTES = 10 * 1024 * 1024  # 10MB
N_COLORS = 3
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# LLM timeout (<= 20s as requested)
DEFAULT_LLM_TIMEOUT_S = 20.0

app = FastAPI()


# ------------------------------------------------------------
# Runtime config (always read from env / loaded .env.*)
# ------------------------------------------------------------
def get_mode() -> str:
    return os.getenv("LLM_MODE", "local").strip().lower()  # local | remote


def get_llm_timeout_s() -> float:
    v = os.getenv("LLM_TIMEOUT_S")
    if not v:
        return DEFAULT_LLM_TIMEOUT_S
    try:
        t = float(v)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_S
    return min(max(t, 1.0), 20.0)  # clamp to [1, 20]


# Local (Ollama)
def get_ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2:1b").strip()


# Remote (OpenAI-compatible Chat Completions)
def get_remote_base_url() -> str:
    return os.getenv("REMOTE_BASE_URL", "https://api.openai.com/v1").strip()


def get_remote_model() -> str:
    return os.getenv("REMOTE_MODEL", "gpt-4o-mini").strip()


def get_remote_api_key() -> str | None:
    k = os.getenv("REMOTE_API_KEY")
    return k.strip() if k else None


# ------------------------------------------------------------
# Middleware: early size check (if Content-Length present)
# ------------------------------------------------------------
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BYTES:
            return JSONResponse(status_code=413, content={"detail": "File too large. Max 10MB."})
        return await call_next(request)


app.add_middleware(LimitUploadSizeMiddleware)


# ------------------------------------------------------------
# Helpers: colors + image metadata
# ------------------------------------------------------------
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
    """
    Validates that bytes represent a real image (Pillow.verify()),
    then extracts:
      - width, height
      - top N dominant colors (hex + weight) via Pillow quantize
    """
    try:
        with Image.open(BytesIO(image_bytes)) as im_check:
            im_check.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid file: not a recognized image format")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file: corrupted or unreadable image")

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


# ------------------------------------------------------------
# LLM: local (Ollama) / remote (OpenAI-compatible)
# ------------------------------------------------------------
def _httpx_timeout_20s() -> httpx.Timeout:
    # total timeout <= 20s; connect timeout shorter to fail fast if not reachable
    total = get_llm_timeout_s()
    return httpx.Timeout(total, connect=min(5.0, total))


async def generate_local(prompt: str) -> tuple[str, str]:
    url = get_ollama_url()
    model = get_ollama_model()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 60},
    }

    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout_20s()) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip(), model

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Local LLM timeout (<=20s)")

    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Cannot reach local model (Ollama). Is it running?")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Local model error: {e.response.text}")


async def generate_remote(prompt: str) -> tuple[str, str]:
    base_url = get_remote_base_url()
    model = get_remote_model()
    api_key = get_remote_api_key()

    if not api_key:
        raise HTTPException(status_code=500, detail="REMOTE_API_KEY is not set (required for LLM_MODE=remote)")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Sei un assistente che converte metadati in una sola frase."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 80,
    }

    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout_20s()) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Remote LLM timeout (<=20s)")

    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Cannot reach remote LLM provider")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Remote LLM error: {e.response.text}")

    try:
        text = data["choices"][0]["message"]["content"].strip()
        return text, model
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

    raise HTTPException(status_code=500, detail=f"Invalid LLM_MODE: {mode} (use 'local' or 'remote')")


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
class DescribeResponse(BaseModel):
    description: str
    mode: Literal["local", "remote"]
    model: str


@app.post("/images/describe", response_model=DescribeResponse)
async def upload_image_and_describe(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type (expected jpeg/png/webp)")

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

    width, height, dominant_colors = extract_metadata_with_pillow(bytes(buf), N_COLORS)
    prompt = build_prompt(width, height, dominant_colors)
    description, mode_used, model_used = await generate_description(prompt)

    return DescribeResponse(description=description, mode=mode_used, model=model_used)