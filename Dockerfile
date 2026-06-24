FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assuming main.py runs a web server or worker. If there's a specific port, we expose it.
# ms-ia generally uses NATS or a REST API on a port like 8000
EXPOSE 8000

CMD ["python", "main.py"]
