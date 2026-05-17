import uuid

from django.db import models


class MicroscopyImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to="uploads/microscopy/")
    treatment_group = models.CharField(max_length=255, blank=True, default="")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    num_channels = models.IntegerField(null=True, blank=True)
    z_slices = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.original_filename


class AnalysisRun(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    progress = models.IntegerField(default=0)
    progress_message = models.CharField(max_length=255, blank=True, default="")
    error_message = models.CharField(max_length=500, blank=True, default="")
    error_traceback = models.TextField(blank=True, default="")
    checkpoint_path = models.CharField(max_length=500, blank=True, default="")
    config_snapshot = models.JSONField(default=dict, blank=True)
    image_manifest = models.JSONField(default=list, blank=True)
    commit_sha = models.CharField(max_length=40, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Run {str(self.id)[:8]} — {self.status}"

    @property
    def expected_image_count(self) -> int:
        return len(self.image_manifest or [])


class ImageResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis_run = models.ForeignKey(
        AnalysisRun, on_delete=models.CASCADE, related_name="results"
    )
    image = models.ForeignKey(
        MicroscopyImage, on_delete=models.CASCADE, related_name="results"
    )
    puncta_count = models.IntegerField(null=True, blank=True)
    puncta_density = models.FloatField(null=True, blank=True)
    mip_image = models.ImageField(upload_to="results/mip/", blank=True)
    segmentation_mask = models.ImageField(upload_to="results/masks/", blank=True)
    overlay_image = models.ImageField(upload_to="results/overlays/", blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["analysis_run", "image"],
                name="uniq_result_per_run_image",
            )
        ]

    def __str__(self):
        return f"Result for {self.image.original_filename}"

