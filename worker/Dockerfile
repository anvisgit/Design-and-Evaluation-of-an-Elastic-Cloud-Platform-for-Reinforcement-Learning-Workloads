FROM python:3.11-slim
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential swig libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY worker/requirements.txt .
RUN pip install -r requirements.txt
COPY lambdas/common/config_schema.py /app/common/config_schema.py
RUN touch /app/common/__init__.py
COPY worker/*.py /app/
RUN useradd -m rl && chown -R rl /app
USER rl
CMD ["python", "main.py"]
