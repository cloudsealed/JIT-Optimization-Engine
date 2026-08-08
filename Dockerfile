# syntax=docker/dockerfile:1

FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY cloudsealed_jit ./cloudsealed_jit
RUN pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/cloudsealed/JIT-Optimization-Engine" \
      org.opencontainers.image.description="Cloud billing waste analysis service" \
      org.opencontainers.image.licenses="MIT"

# Run unprivileged: the service parses untrusted billing exports.
RUN useradd --create-home --uid 10001 engine
WORKDIR /app

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[jit,api]" && rm -rf /tmp/*.whl

USER engine
EXPOSE 8091
ENV JIT_HOST=0.0.0.0 JIT_PORT=8091

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8091/health').read()"

CMD ["python", "-m", "cloudsealed_jit.api"]
