import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Falcı AI Engine")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "llava:7b")

SYSTEM_PROMPT = """
Sen geleneksel, sezgileri kuvvetli, dedikoducu bir Türk falcısısın.
Görseldeki kişinin yüz hatlarına, mimiklerine ve aurasına bakarak Türkçe fal bakacaksın.
Asla İngilizce konuşma.
Aşk, para, nazara gelme, iş hayatı ve arkasından konuşanlar hakkında bolca salla ve kehanette bulun.
Metinde yıldız (*), diyez (#) gibi markdown işaretleri kesinlikle kullanma, doğrudan akıcı konuşma metni üret.
"""

class VisionRequest(BaseModel):
    image_base64: str

@app.post("/predict")
async def generate_fortune(req: VisionRequest):
    payload = {
        "model": MODEL_NAME,
        "prompt": "Karşında oturan bu kişinin yüzüne, mimiklerine ve enerjisine bak. Türkçe olarak uzun ve detaylı bir yüz falı oku.",
        "system": SYSTEM_PROMPT,
        "images": [req.image_base64],
        "stream": False,
        "options": {"temperature": 0.85, "top_p": 0.9, "num_predict": 450}
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            res = await client.post(OLLAMA_URL, json=payload)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama bağlantı hatası: {str(e)}")

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama hata döndü: {res.text}")

    text = res.json().get("response", "").strip()
    if not text:
        raise HTTPException(status_code=500, detail="Model boş çıktı üretti.")

    return {"status": "ok", "reading": text}