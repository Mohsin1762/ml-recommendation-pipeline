FROM python:3.10-slim

WORKDIR /app

# Copy requirements (create requirements.txt with all your deps)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your app source code
COPY app ./app

# Copy saved models (create the folder models and copy your pkl files)
COPY app/models ./models

# Expose port for FastAPI
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
