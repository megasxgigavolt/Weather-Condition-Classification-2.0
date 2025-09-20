# Dockerfile
FROM python:3.11-slim

# System deps (for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deps first to leverage layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY app.py /app/app.py

# Streamlit defaults
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Disable usage stats (optional)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080
CMD ["bash", "-lc", "streamlit run /app/app.py --server.address=${STREAMLIT_SERVER_ADDRESS} --server.port=${STREAMLIT_SERVER_PORT}"]
