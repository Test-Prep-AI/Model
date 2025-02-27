ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION} as base
WORKDIR /app
COPY ./requirements.txt /code/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
COPY . .
EXPOSE 8000
RUN mkdir -p /code/.cache
EXPOSE 80
CMD uvicorn main:app --reload --host 0.0.0.0 --port 8000 --proxy-headers