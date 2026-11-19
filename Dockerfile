FROM python:3.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt
COPY main.py train_model.py model.pkl ./
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
