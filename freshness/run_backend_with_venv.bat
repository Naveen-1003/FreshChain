@echo off
echo 🚀 Starting Food Traceability System with AI Analytics
echo ================================================

cd /d "C:\Users\K R ARAVIND\OneDrive\Desktop\freshness"

echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

echo 🔧 Installing dependencies...
pip install fastapi uvicorn asyncpg passlib python-jose bcrypt python-multipart pillow numpy tensorflow qrcode[pil] transformers torch accelerate

echo 🤖 Starting backend server with AI analytics...
cd hyperledger-food-traceability\api
python -m uvicorn complete_backend:app --reload --host 0.0.0.0 --port 8000

pause