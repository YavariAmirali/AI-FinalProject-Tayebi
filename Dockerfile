# استفاده از نسخه سبک پایتون
FROM python:3.9-slim

# جلوگیری از نوشتن فایل‌های pyc و بافر شدن خروجی‌ها (برای لاگ بهتر)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# کپی و نصب نیازمندی‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8501

# اجرای برنامه
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]