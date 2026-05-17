"""Flip orphaned ``running`` AnalysisRun rows to ``failed``.

After a server restart the in-process executor loses any pending or
running jobs. Their AnalysisRun rows otherwise stay in ``running``
forever. Run this command on deploy / after a crash:

    python manage.py recover_stale_runs

Only rows whose ``started_at`` is older than
``settings.ANALYSIS_STALE_AFTER_MINUTES`` (default 60) are touched, so
genuinely live jobs are left alone.
"""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from synapse_web.models import AnalysisRun


class Command(BaseCommand):
    help = "Mark stale 'running' analysis runs as 'failed' after a restart."

    def add_arguments(self, parser):
        parser.add_argument(
            "--older-than",
            type=int,
            default=settings.ANALYSIS_STALE_AFTER_MINUTES,
            help="Minimum age in minutes for a running row to be considered stale.",
        )

    def handle(self, *args, older_than: int, **opts):
        cutoff = timezone.now() - timedelta(minutes=older_than)
        stale = AnalysisRun.objects.filter(
            status="running", started_at__lt=cutoff
        )
        count = stale.update(
            status="failed",
            finished_at=timezone.now(),
            error_message="Server restart — job orphaned.",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Recovered {count} stale run(s) older than {older_than} min."
            )
        )
