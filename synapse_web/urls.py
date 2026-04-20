from django.urls import path
from . import views

app_name = "synapse_web"

urlpatterns = [
    path("upload/", views.upload, name="upload"),
    path("run/", views.run_inference, name="run_inference"),
    path("results/", views.results, name="results"),
]
