"""
Django settings for synapse_config project.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-only-change-in-production",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "home",
    "synapse_web",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "synapse_config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "synapse_config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db" / "db.sqlite3",
        "OPTIONS": {"timeout": 20},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

DATA_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 10000  # folder uploads can contain many .ets tiles

THESIS_PDF_URL = os.environ.get(
    "THESIS_PDF_URL",
    "https://example.com/anokhina-bachelor-thesis.pdf",
)

ANALYSIS_WORK_FN = os.environ.get(
    "ANALYSIS_WORK_FN",
    "synapse_web.services.ml_jobs.do_work",
)

ANALYSIS_JOBS_SYNC = os.environ.get("ANALYSIS_JOBS_SYNC", "0") == "1"

ANALYSIS_STALE_AFTER_MINUTES = int(
    os.environ.get("ANALYSIS_STALE_AFTER_MINUTES", "60")
)

# ML pipeline I/O locations.
# Override any of these via env vars when deploying (e.g. point
# CHECKPOINT_DIR at a shared NFS mount). Defaults live under MEDIA_ROOT
# so a fresh checkout works without any configuration.
CHECKPOINT_DIR = Path(
    os.environ.get("CHECKPOINT_DIR", MEDIA_ROOT / "checkpoints")
)
PATCHES_UPLOAD_DIR = Path(
    os.environ.get("PATCHES_UPLOAD_DIR", MEDIA_ROOT / "patches_upload")
)
BUNDLES_DIR = Path(
    os.environ.get("BUNDLES_DIR", MEDIA_ROOT / "bundles")
)
RUNS_DIR = Path(
    os.environ.get("RUNS_DIR", MEDIA_ROOT / "runs")
)

# Upload-size cap for encoder checkpoints (.pt). 500 MB by default.
MAX_CHECKPOINT_UPLOAD_BYTES = int(
    os.environ.get("MAX_CHECKPOINT_UPLOAD_BYTES", str(500 * 1024 * 1024))
)

# Run-creation ZIP upload limits.
MAX_ZIP_UPLOAD_BYTES = int(
    os.environ.get("MAX_ZIP_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))
)
MAX_ZIP_UNCOMPRESSED_BYTES = int(
    os.environ.get("MAX_ZIP_UNCOMPRESSED_BYTES", str(20 * 1024 * 1024 * 1024))
)
MAX_ZIP_MEMBERS = int(os.environ.get("MAX_ZIP_MEMBERS", "200000"))
