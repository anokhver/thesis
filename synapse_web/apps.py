from django.apps import AppConfig


class SynapseWebConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "synapse_web"

    def ready(self) -> None:
        from django.conf import settings

        from . import signals  # noqa: F401  (registers the signal handlers)

        for attr in ("CHECKPOINT_DIR", "RUNS_DIR"):
            path = getattr(settings, attr, None)
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
