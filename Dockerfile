FROM python:3.11-slim

# System deps (optional but helpful for scientific wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port","8501","--server.address","0.0.0.0"]
