"""Filename parsing for microscopy uploads.

The dataset follows the convention `<session>_<replicate>_<GROUP>.<ext>`,
e.g. `1_8_KONTROLA.vsi`, `2_3_PSYHARMIN.vsi`. The GROUP token is the
sample-net name copied verbatim from the cellSens `.oex` acquisition plan.
The full convention is documented in the thesis text, chapter "Data".

Add new tokens to GROUP_TREATMENTS as the dataset grows.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

GROUP_TREATMENTS: dict[str, str] = {
    "KONTROLA": "Control",
    "PSI": "Psilocin",
    "PSY": "Psilocybin",
    "PSYHARMIN": "Psilocybin + Harmine",
    "NORPSI": "Norpsilocin",
    "NORPSIHARMIN": "Norpsilocin + Harmine",
    "BAEO": "Baeocystin",
    "BAEOHARMIN": "Baeocystin + Harmine",
}

KNOWN_GROUPS: tuple[str, ...] = tuple(GROUP_TREATMENTS.keys())

_TOKEN_RE = re.compile(r"[A-Z0-9]+")


def _basename(filename: str) -> str:
    name = PureWindowsPath(filename).name
    name = PurePosixPath(name).name
    return name


def _tokens(filename: str) -> list[str]:
    base = _basename(filename)
    stem = base.split(".", 1)[0]
    return _TOKEN_RE.findall(stem.upper())


def derive_treatment_group(filename: str) -> str | None:
    """Return the GROUP token from a microscopy filename, or None.

    Matching is exact on whole tokens after splitting on non-alphanumeric
    characters, so `NORPSI` does not accidentally match `PSI` and
    `BAEOHARMIN` does not get truncated to `BAEO`.
    """
    known = set(KNOWN_GROUPS)
    for tok in _tokens(filename):
        if tok in known:
            return tok
    return None
