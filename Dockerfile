FROM python:3.12-slim

WORKDIR /app

# System deps for matplotlib, pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure output & data dirs exist
RUN mkdir -p outputs data

EXPOSE 7080

CMD ["python", "app.py"]
