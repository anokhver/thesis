# Multi-stage build for the Synaptic Cluster Explorer container.
# Stage 1 installs the Python wheels + builds the synaptic_ssl package.
# Stage 2 copies only the venv + source needed at runtime.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel

# Web stack + clustering deps. Pinned in requirements/web.txt so the
# build is deterministic.
COPY requirements/web.txt requirements/web.txt
RUN pip install -r requirements/web.txt

# Install the synaptic_ssl package. We only need pyproject.toml + the
# source under src/ + README (referenced by pyproject) — this keeps
# the layer cache valid as long as the package code is unchanged.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install .


# ---------------------------------------------------------------------------
# Runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# OCI image annotations for Docker Hub / GHCR provenance.
LABEL org.opencontainers.image.title="Synaptic Cluster Explorer" \
      org.opencontainers.image.description="Clustering analysis UI for the SynAPSeg self-supervised microscopy pipeline" \
      org.opencontainers.image.authors="Veronika Anokhina <veronika.i.anokhina@gmail.com>" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="0.1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=synapse_config.settings

# Copy the wheel-installed venv from the builder stage. No build tools
# end up in the runtime image, so it stays a few hundred MB lighter.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Persistent dirs (also volume-mountable). Created here so the image
# works standalone without compose volumes too.
RUN mkdir -p /app/db /app/media /app/staticfiles

# collectstatic must run with DEBUG=0 so the production manifest storage
# generates style.HASH.css entries WhiteNoise can serve with far-future
# cache headers. DJANGO_DEBUG only affects this build step; runtime
# inherits whatever the operator sets (defaults to "True" in settings).
RUN DJANGO_DEBUG=0 python manage.py collectstatic --noinput

# Drop privileges. The named uid/gid (10001) makes it easy for hosts to
# pre-create directories with matching ownership for bind-mounted volumes.
RUN groupadd -r -g 10001 app \
    && useradd -r -u 10001 -g app -d /app -s /sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz/', timeout=3).status == 200 else 1)"

# gunicorn with one process so the in-process job executor's single
# worker contract (services/jobs.py) is preserved, plus 4 threads so
# status polling stays responsive while a clustering run is busy.
# --timeout 1800 covers long extract+cluster runs on CPU (~10+ min on
# real datasets); torch and numpy release the GIL for compute so polling
# threads still serve status JSON during a run.
CMD ["gunicorn", "synapse_config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "1800", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
