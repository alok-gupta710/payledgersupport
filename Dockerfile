# WHY python:3.11-slim: slim removes build tools we don't need → 200MB smaller
FROM python:3.11-slim

# WHY /app: standard convention for app code inside containers
WORKDIR /app

# WHY copy requirements.txt BEFORE app.py:
# Docker caches each build step. If requirements.txt hasn't changed,
# Docker reuses the cached pip install layer → much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# WHY EXPOSE 8080: tells Cloud Run which port to route traffic to
EXPOSE 8000

# WHY CMD not ENTRYPOINT: Cloud Run can override CMD at runtime if needed
CMD ["python", "app.py"]
