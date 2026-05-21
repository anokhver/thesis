from django.urls import path

from . import views

app_name = "synapse_web"

urlpatterns = [
    path("runs/<uuid:run_id>/", views.run_detail, name="run_detail"),
    path("runs/<uuid:run_id>/status/", views.run_status, name="run_status"),
    path("results/", views.results, name="results"),
    path("overview/", views.overview, name="overview"),
]
