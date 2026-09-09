import base64
import json
import os
import subprocess
import sys
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
Sen kameradan karşına oturan kişinin tipine, yüz ifadesine, aurasına ve enerjisine bakarak fal bakan, dedikoducu, hafif dobra ve patavatsız bir mahalle falcısısın.
Kişinin yüzündeki yorgunluktan, gözlerinden, duruşundan ya da giyiminden yola çıkarak akıl almaz çıkarımlar yap. 
Geçmişi, geleceği, arkasından konuşanları, aşk hayatındaki tıkanıklıkları ve kafasındaki tilkileri bolca sallayarak yüzüne vur.
'Yüzünde bir ağırlık var...', 'Omuzlarına birileri yük bindirmiş...', 'Gözünün feri gitmiş senin, kim üzdü seni?' gibi mistik ve drama dozu yüksek cümleler kur.
Asla 'bu bir fotoğraf' veya 'görüntü kalitesi düşük' deme; tamamen mistik bir enerji okuması yapıyormuş gibi kesin konuş.
"""

def capture_frame():
    img_path = "/tmp/kamera_kare.jpg"
    print("Kamera açılıyor... Ekrana bak...")
    
    # ffmpeg ile v4l2 üzerinden tek kare yakala
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "v4l2",
        "-video_size", "1280x720",
        "-i", "/dev/video0",
        "-frames:v", "1",
        "-loglevel", "quiet",
        img_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Hata: ffmpeg ile görüntü alınamadı: {e}")
        sys.exit(1)

    if not os.path.exists(img_path):
        print("Hata: Fotoğraf dosyası kaydedilemedi.")
        sys.exit(1)

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
        
    os.remove(img_path)
    return b64

def fal_bak(b64_image):
    payload = {
        "model": "qwen2-vl:2b",
        "prompt": "Gözlerimin içine, yüzümün çizgilerine ve duruşuma bak. Enerjimde ne görüyorsun? Döke saça anlat halimi.",
        "system": SYSTEM_PROMPT,
        "images": [b64_image],
        "stream": True,
        "options": {
            "temperature": 0.95,
            "top_p": 0.95,
            "num_predict": 600
        }
    }

    print("\nFalcı auranı süzüyor, arkana yaslan...\n" + "-" * 40)
    
    with httpx.Client(timeout=90.0) as client:
        with client.stream("POST", OLLAMA_URL, json=payload) as response:
            if response.status_code != 200:
                print(f"Hata: Ollama yanıt vermedi ({response.status_code})")
                return
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    print(chunk.get("response", ""), end="", flush=True)
    print("\n" + "-" * 40)

if __name__ == "__main__":
    b64_img = capture_frame()
    fal_bak(b64_img)