import os
import io
import re
import asyncio
import base64
from typing import Optional

from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="Falcı AI Engine", docs_url=None, redoc_url=None)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY yok; container'ı boşuna ayağa kaldırma.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini 3 varsayılanı HIGH thinking; taslak ("Brainstorming / Opening") fal diye sızmasın diye düşünmeyi kısarız.
SYSTEM_PROMPT = """
Sen mahallenin dobra, hazırcevap Türk falcısısın. Karşındakine yüksek sesle fal bakıyorsun.

KESİN:
- Çıktın SADECE falcının ağzından çıkan Türkçe konuşmadır.
- Plan, taslak, analiz, başlık, madde, İngilizce, markdown, "Opening", "Brainstorming" YASAK.
- Fiziksel tarif YASAK: burun, alın, çene, ağız, kaş, yüz şekli, göz aralığı, kilo, saç, cilt, güzellik, çirkinlik, "yuvarlak/oval/uzun yüz" yok.
- Fotoğrafa bakıp görünüşü anlatma. Aşk, para, karakter ve kısmet konuş.
- Doğrudan konuşmaya başla. 4-5 tam cümle. Hastalık/ölüm yok.
"""

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_IMAGE_BYTES = 400_000
MAX_OUTPUT_TOKENS = 700
LEAK_RE = re.compile(
    r"(brainstorm|opening\s*:|based on features|ideas based|outline|\*\*|#{1,3}\s)",
    re.IGNORECASE,
)
# Stand'da insanı teşhir etmesin diye görünüşü fal metninden ele.
APPEARANCE_RE = re.compile(
    r"(geniş burun|ince burun|burnun |burunun |"
    r"alçak alın|geniş alın|"
    r"sivri çene|çenen |çenenin |"
    r"oval yüz|yuvarlak yüz|uzun yüz|yüz şekl|yüz formu|"
    r"küçük ağız|geniş ağız|göz aral|bakışlı|"
    r"kilolu|şişman|tombul|sivilce|kırışık)",
    re.IGNORECASE,
)

# Ölçümü modele "geniş burun" diye vermiyoruz; fal temasına çeviriyoruz.
FEATURE_THEMES = {
    "yuvarlak": "sıcakkanlı ve sosyalsin",
    "oval": "dengeli ve uyumlusun",
    "uzun": "içinden düşünüp taşınıyorsun",
    "alçak alın": "pratik ve acelecisin",
    "orta alın": "kafan dengeli çalışıyor",
    "geniş alın": "planlısın, kafa yoruyorsun",
    "yakın bakışlı": "detaycısın, kolay kanmazsın",
    "dengeli göz aralığı": "duyguların net okunuyor",
    "geniş bakışlı": "hayalcisin, uzağı kuruyorsun",
    "ince burun": "parada titizsin",
    "dengeli burun": "paran da hayatın da dengede",
    "geniş burun": "cömertsin, kapın açık",
    "küçük ağız": "içine atarsın, aşkta seçicisin",
    "dengeli ağız": "açık sözlüsün",
    "geniş ağız": "dilin açık, aşkın dilinden kaçıyor",
    "sivri çene": "inatçısın, kararından dönmezsin",
    "dengeli çene": "esneksin ama dik durursun",
    "geniş çene": "kafa tutarsın, iraden sert",
    "hafif gülümseme": "için şu an rahat",
    "nötr ifade": "bir şey saklıyorsun",
}


class FaceFeatures(BaseModel):
    face_shape: str = ""
    forehead: str = ""
    eye_spacing: str = ""
    nose: str = ""
    mouth: str = ""
    jaw: str = ""
    expression: str = ""
    gaze: str = ""


class VisionRequest(BaseModel):
    image_base64: str = Field(..., min_length=32, max_length=700_000)
    face_features: Optional[FaceFeatures] = None


def _decode_face_image(raw_b64: str) -> Image.Image:
    img_str = raw_b64.split(",", 1)[1] if "," in raw_b64 else raw_b64
    try:
        img_bytes = base64.b64decode(img_str, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.") from exc

    if len(img_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Görsel çok büyük.")

    try:
        image = Image.open(io.BytesIO(img_bytes))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Geçersiz görsel.") from exc

    if image.format not in ("JPEG", "JPG", "WEBP", "PNG"):
        raise HTTPException(status_code=400, detail="Desteklenmeyen görsel.")

    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((448, 448))
    return image


def _themes_from_features(features: Optional[FaceFeatures]) -> list[str]:
    if features is None:
        return []
    themes = []
    seen = set()
    for value in (
        features.face_shape,
        features.forehead,
        features.eye_spacing,
        features.nose,
        features.mouth,
        features.jaw,
        features.expression,
    ):
        value = (value or "").strip().lower()
        theme = FEATURE_THEMES.get(value)
        if theme and theme not in seen:
            seen.add(theme)
            themes.append(theme)
    return themes[:4]


def _build_user_prompt(features: Optional[FaceFeatures], strict: bool = False) -> str:
    if strict:
        return (
            "Fal bak. Sadece falcının söylediği Türkçe cümleleri yaz. "
            "Yüzünü, burnunu, çenesini, alnını, kilosunu asla anlatma."
        )

    themes = _themes_from_features(features)
    if themes:
        return (
            "Karşındakine fal bak. Görünüşünü tarif etme. "
            f"Şu karakter/kısmet izlenimlerini doğallaştır: {'; '.join(themes)}. "
            "Şimdi falcı olarak konuşmaya başla."
        )
    return "Karşındakine fal bak. Görünüşünü tarif etme. Şimdi falcı olarak konuşmaya başla."


def _thinking_config():
    thinking_level = getattr(types, "ThinkingLevel", None)
    if thinking_level is not None:
        level = getattr(thinking_level, "MINIMAL", None) or getattr(thinking_level, "LOW", None)
        if level is not None:
            return types.ThinkingConfig(thinking_level=level, include_thoughts=False)
    return types.ThinkingConfig(thinking_budget=0, include_thoughts=False)


def _gen_config() -> types.GenerateContentConfig:
    kwargs = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.7,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "text/plain",
    }
    try:
        kwargs["thinking_config"] = _thinking_config()
    except Exception as exc:
        print(f"thinking_config atlandı: {exc}", flush=True)
    return types.GenerateContentConfig(**kwargs)


def _visible_text(response) -> str:
    chunks = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks).strip()
    try:
        return (response.text or "").strip()
    except Exception:
        return ""


def _is_spoken_fortune(text: str) -> bool:
    if len(text) < 40:
        return False
    if LEAK_RE.search(text) or APPEARANCE_RE.search(text):
        return False
    stripped = text.lstrip()
    if stripped.startswith(("*", "#", "-", "•")):
        return False
    return True


def _call_gemini(image: Image.Image, prompt: str, use_thinking_config: bool = True):
    config = _gen_config() if use_thinking_config else types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="text/plain",
    )
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=[image, prompt],
        config=config,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def generate_fortune(req: VisionRequest):
    image = _decode_face_image(req.image_base64)
    last_error = None
    use_thinking_config = True

    for attempt in range(2):
        prompt = _build_user_prompt(req.face_features, strict=(attempt == 1))
        try:
            response = await asyncio.to_thread(_call_gemini, image, prompt, use_thinking_config)
            reading = _visible_text(response)
            if _is_spoken_fortune(reading):
                return {"status": "ok", "reading": reading}
            last_error = f"taslak sızıntısı: {reading[:80]!r}"
            print(f"Gemini deneme {attempt + 1} reddedildi: {last_error}", flush=True)
        except Exception as exc:
            last_error = exc
            print(f"Gemini deneme {attempt + 1} hata: {exc}", flush=True)
            msg = str(exc).lower()
            if "thinking" in msg:
                use_thinking_config = False
        if attempt == 0:
            await asyncio.sleep(0.4)

    print(f"Gemini başarısız: {last_error}", flush=True)
    raise HTTPException(status_code=502, detail="Falcı şu an dalgın, bir daha dene.")
