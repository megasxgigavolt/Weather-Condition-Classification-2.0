# Python 3.11 slim + xgboost runtime dep
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgomp1 is required by xgboost; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for build caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Streamlit on 0.0.0.0:8080 (Beanstalk’s nginx will proxy this)
ENV PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 8080

# Simple healthcheck; Streamlit exposes a health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
