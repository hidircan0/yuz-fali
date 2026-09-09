import asyncio
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import edge_tts
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://falci-ai-engine:8000/predict")
AUDIO_STORAGE_DIR = os.getenv("AUDIO_STORAGE_DIR", "/shared/audio")
VOICE = "tr-TR-EmelNeural"
AUDIO_TTL_SEC = int(os.getenv("AUDIO_TTL_SEC", "180"))
SAFE_AUDIO_NAME = re.compile(r"^[a-f0-9]{32}\.mp3$")

http_client: Optional[httpx.AsyncClient] = None
state_lock = asyncio.Lock()
is_busy = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    yield
    await http_client.aclose()
    http_client = None


app = FastAPI(title="Falcı API Gateway", version="1.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class FaceFeatures(BaseModel):
    face_shape: str = ""
    forehead: str = ""
    eye_spacing: str = ""
    nose: str = ""
    mouth: str = ""
    jaw: str = ""
    expression: str = ""
    gaze: str = ""


class FortuneRequest(BaseModel):
    image_base64: str = Field(..., min_length=32, max_length=700_000)
    face_features: Optional[FaceFeatures] = None


class FortuneResponse(BaseModel):
    reading: str
    audio_url: str


def _purge_old_audio() -> None:
    now = time.time()
    try:
        names = os.listdir(AUDIO_STORAGE_DIR)
    except OSError:
        return
    for name in names:
        if not name.endswith(".mp3"):
            continue
        path = os.path.join(AUDIO_STORAGE_DIR, name)
        try:
            if now - os.path.getmtime(path) > AUDIO_TTL_SEC:
                os.remove(path)
        except OSError:
            pass


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/fortune", response_model=FortuneResponse)
async def handle_fortune(payload: FortuneRequest):
    global is_busy
    async with state_lock:
        if is_busy:
            raise HTTPException(status_code=429, detail="Falcı birine bakıyor, sıranı bekle.")
        is_busy = True

    try:
        _purge_old_audio()

        body = {"image_base64": payload.image_base64}
        if payload.face_features is not None:
            body["face_features"] = payload.face_features.model_dump()

        try:
            ai_res = await http_client.post(AI_SERVICE_URL, json=body)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Falcı henüz uyanmadı, birkaç saniye sonra dene.")
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Falcı uzattı, bir daha dene.")

        if ai_res.status_code != 200:
            raise HTTPException(status_code=502, detail="Falcı şu an dalgın, bir daha dene.")

        try:
            reading_text = (ai_res.json().get("reading") or "").strip()
        except Exception:
            raise HTTPException(status_code=502, detail="Falcı şu an dalgın, bir daha dene.")

        if not reading_text:
            raise HTTPException(status_code=502, detail="Falcı mırıldandı, bir daha dene.")

        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_STORAGE_DIR, audio_filename)

        try:
            communicator = edge_tts.Communicate(reading_text, VOICE)
            await communicator.save(audio_path)
        except Exception as exc:
            print(f"TTS Hatası: {exc}", flush=True)
            return FortuneResponse(reading=reading_text, audio_url="")

        return FortuneResponse(
            reading=reading_text,
            audio_url=f"/api/v1/audio/{audio_filename}",
        )
    finally:
        is_busy = False


@app.get("/api/v1/audio/{filename}")
async def stream_audio(filename: str):
    if not SAFE_AUDIO_NAME.fullmatch(filename):
        raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı.")
    file_path = os.path.join(AUDIO_STORAGE_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı.")
    return FileResponse(file_path, media_type="audio/mpeg")
