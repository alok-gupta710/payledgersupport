# WHY python:3.10-slim: slim removes build tools we don't need → 200MB smaller
FROM python:3.10-slim

# WHY /app: standard convention for app code inside containers
WORKDIR /app

# WHY copy requirements.txt BEFORE app.py:
# Docker caches each build step. If requirements.txt hasn't changed,
# Docker reuses the cached pip install layer → much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY service.py .
COPY config.py .
COPY observability.py .
COPY resilience.py .
COPY llm_agent.py .
COPY rag_agent.py .
COPY tool_agent.py .
COPY memory_agent.py .
COPY adaptive_agent.py .
COPY baseline_agent.py .
COPY healthcheck.py .
COPY scenarios.py .
COPY evaluation.py .
COPY run_eval.py .
COPY safety_review.py .
COPY data/ ./data/
COPY prompts/ ./prompts/
COPY eval/ ./eval/
COPY results/ ./results/
COPY VERSION .

# WHY EXPOSE 8501: tells Cloud Run which port to route traffic to
EXPOSE 8501

# WHY CMD not ENTRYPOINT: Cloud Run can override CMD at runtime if needed
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
