FROM python:3.12-slim

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

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
