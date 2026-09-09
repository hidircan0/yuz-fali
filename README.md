# Mistik Yüz Falcısı

Stand için eğlence zıkkımı. Webcam’den yüz kilitlenir, kısa bir Türkçe fal çıkar, sesli okunur. Gerçek fizyonomi analizi değil; ölçülen hatlar içeride karakter/kısmet temasına çevrilir, falcı yüzü tarif etmez.

## Gereksinimler

- Docker ve Docker Compose
- Google Gemini API anahtarı
- İnternet (Gemini + ilk açılışta yüz modeli CDN)

## Çalıştırma

Proje kökünde `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Sonra:

```bash
docker compose up --build
```

Tarayıcı: [http://localhost:39842](http://localhost:39842)

Kamera izni ver. Yüz ovalde yeşil olunca **Falıma Bak**.

## Stand notu

- Aynı anda tek fal; butona spam kota yakmaz.
- Yüz kilitlenmeden istek gitmez.
- Ses dosyaları birkaç dakikada silinir.
- Bu bir fuar/stand oyuncağı; internete açık servis diye düşünme.

## Mimari

```
tarayıcı  →  gateway (:39842)  →  ai-engine (iç ağ)  →  Gemini
                 ↓
              Edge TTS
```

| Servis | Ne işe yarar |
|--------|----------------|
| `gateway` | Arayüz, kuyruk kilidi, TTS |
| `ai-engine` | Görsel + fal üretimi |

## Durdurma

```bash
docker compose down
```
