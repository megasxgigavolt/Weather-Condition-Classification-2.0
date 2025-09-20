# Python 3.11 with slim base
FROM python:3.11-slim

# System libs (libgomp is needed by xgboost wheel)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EB’s nginx will map to this container port
ENV PORT=8080
EXPOSE 8080

# Streamlit must listen on 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
