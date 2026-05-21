from django.contrib import admin
from django.http import HttpResponse
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static


def _healthz(_request):
    """Liveness check — intentionally does not touch the DB."""
    return HttpResponse("ok", content_type="text/plain")


urlpatterns = [
    path("healthz/", _healthz, name="healthz"),
    path("admin/", admin.site.urls),
    path("", include("home.urls")),
    path("analysis/", include("synapse_web.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
