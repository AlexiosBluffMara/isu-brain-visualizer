FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git \
      nodejs \
      npm \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml package.json README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY viewer ./viewer
COPY docs ./docs
COPY data ./data

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

EXPOSE 8080

CMD ["python", "-m", "http.server", "8080"]
