from django.contrib import admin

from .models import AnalysisRun, ImageResult, MicroscopyImage

admin.site.register(MicroscopyImage)
admin.site.register(AnalysisRun)
admin.site.register(ImageResult)
