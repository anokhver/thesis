import uuid

from django.db import models


class AnalysisRun(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    RUN_KIND_CHOICES = [
        ("extract_and_cluster", "Extract embeddings + cluster"),
        ("cluster_only", "Cluster only (pre-built bundle)"),
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
    commit_sha = models.CharField(max_length=40, blank=True, default="")

    run_kind = models.CharField(
        max_length=20, choices=RUN_KIND_CHOICES,
        default="extract_and_cluster",
    )
    checkpoint_path = models.CharField(max_length=500, blank=True, default="")
    config_snapshot = models.JSONField(default=dict, blank=True)
    input_manifest = models.JSONField(default=list, blank=True)

    input_dir = models.CharField(max_length=500, blank=True, default="")
    bundle_dir = models.CharField(max_length=500, blank=True, default="")
    output_dir = models.CharField(max_length=500, blank=True, default="")

    n_patches = models.PositiveIntegerField(null=True, blank=True)
    n_source_images = models.PositiveIntegerField(null=True, blank=True)
    n_clusters = models.PositiveIntegerField(null=True, blank=True)
    embedding_dim = models.PositiveIntegerField(null=True, blank=True)
    pca_dim = models.PositiveIntegerField(null=True, blank=True)
    picked_resolution = models.FloatField(null=True, blank=True)
    mean_ari = models.FloatField(null=True, blank=True)

    clustering_summary = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Run {str(self.id)[:8]} — {self.status}"

    @property
    def expected_image_count(self) -> int:
        return len(self.input_manifest or [])


class SourceImageStats(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis_run = models.ForeignKey(
        AnalysisRun, on_delete=models.CASCADE, related_name="source_stats",
    )
    source_image = models.CharField(max_length=500)
    treatment_group = models.CharField(max_length=255, blank=True, default="")
    n_patches = models.PositiveIntegerField()
    cluster_counts = models.JSONField(default=dict)
    dominant_cluster = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["source_image"]
        constraints = [
            models.UniqueConstraint(
                fields=["analysis_run", "source_image"],
                name="uniq_sourcestat_per_run_image",
            )
        ]

    def __str__(self):
        return f"{self.source_image} (run {str(self.analysis_run_id)[:8]})"
