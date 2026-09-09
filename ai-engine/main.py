import os
import io
import base64
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Falcı AI Engine")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
Sen mahallenin en dedikoducu, hazırcevap, dobra Türk falcısısın.
Doğrudan karşındaki kişinin yüzüne bakarak Türkçe fal okuyorsun.

KURALLAR:
1. SADECE falcının ağzından çıkan konuşmayı yaz. Asla İngilizce kelime, düşünme süreci, taslak, analiz, başlık veya madde işareti (*, # vb.) yazma.
2. Doğrudan Türkçe bir açılışla başla (Örnek: 'Amanın...', 'Vay vay vay...', 'Gözlerindeki bu dalgınlık ne böyle...').
3. Karşındakinin aşkı, parası ve arkasından konuşan sinsi arkadaşları hakkında 4-5 akıcı cümle söyle.
4. Cümlelerini asla yarım bırakma; net, tam bir final cümlesiyle bitir.
"""

class VisionRequest(BaseModel):
    image_base64: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def generate_fortune(req: VisionRequest):
    try:
        img_str = req.image_base64
        if "," in img_str:
            img_str = img_str.split(",")[1]
        
        img_bytes = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_bytes))

        # Geçersiz thinking_config kaldırıldı, temiz konfigürasyon:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.8,
            max_output_tokens=1000
        )

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[image, "Yüzüme bak ve doğrudan dobra bir Türkçe fal söyle. Sadece konuşma metnini ver."],
            config=config
        )

        reading = response.text.strip()
        return {"status": "ok", "reading": reading}

    except Exception as e:
        print(f"Hata detayı: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))