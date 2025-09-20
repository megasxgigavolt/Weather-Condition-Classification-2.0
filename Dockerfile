# Python 3.11 slim base
FROM python:3.11-slim

# Needed by xgboost wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# EB’s Nginx will proxy to this container port
ENV PORT=8080
EXPOSE 8080

# Streamlit must bind 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
