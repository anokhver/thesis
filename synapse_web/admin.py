from django.contrib import admin

from .models import AnalysisRun, SourceImageStats

admin.site.register(AnalysisRun)
admin.site.register(SourceImageStats)
