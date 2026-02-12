# استفاده از نسخه سبک پایتون
FROM python:3.9-slim

# تنظیم دایرکتوری کاری داخل کانتینر
WORKDIR /app

# کپی کردن فایل نیازمندی‌ها و نصب آن‌ها
COPY requirements.txt .

# نصب کتابخانه‌های مورد نیاز (بدون ذخیره کش برای کاهش حجم)
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کل فایل‌های پروژه به داخل کانتینر
COPY . .

# باز کردن پورت پیش‌فرض Streamlit
EXPOSE 8501

# دستور اجرای برنامه دمو طبق سند
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]