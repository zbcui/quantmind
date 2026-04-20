FROM python:3.12-slim

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# Store DB + outputs on HF persistent volume so data survives redeploys
ENV KRONOS_TOOLKIT_OUTPUT=/data

WORKDIR /app

# System deps for matplotlib, pyarrow, git (for pip+git installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure output & data dirs exist
RUN mkdir -p /data data

EXPOSE 7080

CMD ["python", "app.py"]
