"""Form classes for the synapse_web app."""
from __future__ import annotations

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError

from .models import AnalysisRun
from .services.checkpoints import list_checkpoints, validate_checkpoint_upload


class CheckpointUploadForm(forms.Form):
    """Upload form for encoder checkpoints (.pt / .pth)."""

    file = forms.FileField(label="Checkpoint file (.pt or .pth)")

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        # Pure validation — no filesystem writes happen here.
        validate_checkpoint_upload(uploaded)
        return uploaded


class NewRunForm(forms.Form):
    """Create a new clustering run from an uploaded archive."""

    run_kind = forms.ChoiceField(
        choices=AnalysisRun.RUN_KIND_CHOICES,
        widget=forms.RadioSelect,
        label="Run mode",
    )
    checkpoint = forms.ChoiceField(
        required=False,
        label="Encoder checkpoint",
        help_text="Required for extract+cluster runs.",
    )
    archive = forms.FileField(
        label="ZIP upload",
        help_text="Patches folder or pre-computed bundle, packaged as .zip.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = [c["name"] for c in list_checkpoints()]
        self.checkpoint_names = names
        self.fields["checkpoint"].choices = [("", "— select —")] + [
            (n, n) for n in names
        ]

    def clean_archive(self):
        f = self.cleaned_data["archive"]
        max_size = int(settings.MAX_ZIP_UPLOAD_BYTES)
        if f.size is not None and f.size > max_size:
            raise ValidationError(
                f"Archive is too large ({f.size} > {max_size} bytes)."
            )
        name = (f.name or "").lower()
        if not name.endswith(".zip"):
            raise ValidationError("Upload must be a .zip file.")
        return f

    def clean_checkpoint(self):
        value = (self.cleaned_data.get("checkpoint") or "").strip()
        if value and value not in self.checkpoint_names:
            raise ValidationError(
                "Selected checkpoint is no longer available; "
                "pick another or re-upload it."
            )
        return value

    def clean(self):
        cleaned = super().clean()
        kind = cleaned.get("run_kind")
        ckpt = cleaned.get("checkpoint")
        if kind == "extract_and_cluster" and not ckpt:
            self.add_error(
                "checkpoint",
                "A checkpoint is required for extract+cluster runs.",
            )
        return cleaned
