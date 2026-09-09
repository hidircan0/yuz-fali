FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Bağımlılıkları önbellekle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodları ve statik dosyaları aktar
COPY static/ ./static/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]