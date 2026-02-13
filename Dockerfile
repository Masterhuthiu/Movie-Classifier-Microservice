FROM python:3.11-slim

# Không tạo .pyc, log ra stdout ngay
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài system deps cần thiết cho build package (motor, httpx, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker layer cache
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Để Python import được module app.*
ENV PYTHONPATH=/app

EXPOSE 8083

# Chạy FastAPI production bằng uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8083"]
