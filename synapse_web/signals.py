"""Central signal-handler import point.

Each service module registers its own ``@receiver`` decorators; this
file just imports them so ``apps.py::ready()`` can wire everything up
with a single ``from . import signals``.
"""
from synapse_web.services import run_artifacts  # noqa: F401
