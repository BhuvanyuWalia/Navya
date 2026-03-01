FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY static/ static/

# Copy model checkpoint (place ddqn_checkpoint.pth in repo root)
COPY ddqn_checkpoint.pth . 2>/dev/null || echo "No checkpoint found — using random weights"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
