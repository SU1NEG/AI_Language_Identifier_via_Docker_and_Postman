# Python 3.9 tabanlı bir image kullanın
FROM python:3.9-slim

# Çalışma dizinini oluşturun ve ayarlayın
WORKDIR /app

# Gerekli paketleri yükleyin
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükleyin
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyalayın
COPY . .

# FastText modelini indirin
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Uygulamayı başlatın
CMD ["python", "app.py"]
