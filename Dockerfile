FROM python:3.11-slim

WORKDIR /app

# Cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Thêm biến môi trường để Python nhận diện thư mục app là một module
ENV PYTHONPATH=/app

EXPOSE 8083

# Chạy app dưới dạng module để xử lý các lệnh import tốt hơn
CMD ["python", "-m", "app.main"]