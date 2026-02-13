FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Port microservice của bạn
EXPOSE 8083

CMD ["python", "main.py"]