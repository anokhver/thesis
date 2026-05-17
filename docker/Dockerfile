FROM python:3.11-slim

ARG INCLUDE_ML=0

WORKDIR /app

COPY requirements_web.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 60 -r requirements_web.txt

# Flip INCLUDE_ML=1 once the real ML pipeline is merged. requirements.txt
# is the place to add torch / monai / aicsimageio so they land in the
# same image.
COPY requirements.txt .
RUN if [ "$INCLUDE_ML" = "1" ]; then \
        pip install --no-cache-dir --retries 10 --timeout 60 -r requirements.txt ; \
    fi

COPY . .

RUN mkdir -p /app/db /app/media

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=synapse_config.settings

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3).status==200 else 1)" \
    || exit 1

