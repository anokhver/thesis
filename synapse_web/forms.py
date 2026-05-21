"""Form classes for the synapse_web app."""
from __future__ import annotations

from django import forms

from .services.checkpoints import validate_checkpoint_upload


class CheckpointUploadForm(forms.Form):
    """Upload form for encoder checkpoints (.pt / .pth)."""

    file = forms.FileField(label="Checkpoint file (.pt or .pth)")

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        # Pure validation — no filesystem writes happen here.
        validate_checkpoint_upload(uploaded)
        return uploaded
