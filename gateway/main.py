import os
import uuid
import httpx
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Falcı API Gateway", version="1.0.0")

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:5001/predict")
AUDIO_STORAGE_DIR = os.getenv("AUDIO_STORAGE_DIR", "/shared/audio")
VOICE = "tr-TR-EmelNeural"

os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)

# Statik arayüz dosyasını ana dizine bağla
app.mount("/app", StaticFiles(directory="static", html=True), name="static")

class FortuneRequest(BaseModel):
    image_base64: str = Field(..., description="JPEG base64 encoded string")

class FortuneResponse(BaseModel):
    reading: str
    audio_url: str

@app.get("/")
async def root():
    # Doğrudan arayüze yönlendir
    return FileResponse("static/index.html")

@app.post("/api/v1/fortune", response_model=FortuneResponse)
async def handle_fortune(payload: FortuneRequest):
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="Görsel verisi boş olamaz.")

    # 1. AI Servisine (ai-service:5001) proxy yap
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            ai_res = await client.post(
                AI_SERVICE_URL,
                json={"image_base64": payload.image_base64}
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="AI servisine bağlanılamadı. Servisin ayakta olduğundan emin olun.")
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="AI servisi zaman aşımına uğradı (model yanıt vermedi).")

    if ai_res.status_code != 200:
        raise HTTPException(status_code=ai_res.status_code, detail=f"AI Engine Hatası: {ai_res.text}")

    reading_text = ai_res.json().get("reading", "").strip()
    if not reading_text:
        raise HTTPException(status_code=500, detail="Model boş çıktı üretti.")

    # 2. TTS ile sesi paylaşılan volume'e kaydet
    file_id = f"{uuid.uuid4().hex}"
    audio_filename = f"{file_id}.mp3"
    audio_path = os.path.join(AUDIO_STORAGE_DIR, audio_filename)

    try:
        communicator = edge_tts.Communicate(reading_text, VOICE)
        await communicator.save(audio_path)
    except Exception as e:
        # TTS patlasa bile fal metnini kullanıcıya döndür
        return FortuneResponse(reading=reading_text, audio_url="")

    return FortuneResponse(
        reading=reading_text,
        audio_url=f"/api/v1/audio/{audio_filename}"
    )

@app.get("/api/v1/audio/{filename}")
async def stream_audio(filename: str):
    file_path = os.path.join(AUDIO_STORAGE_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı.")
    return FileResponse(file_path, media_type="audio/mpeg")