"""Treatment-group classification from source-image filenames.

Multi-stage matching loaded from a JSON config so the same mapping is
shared between the notebook and CLI extractors:

1. *Base pattern* (``patterns`` list, first-match-wins, substring):
   the first entry whose substring appears in the UPPER-cased
   filename decides the base treatment label. Order is significant —
   place more specific patterns before more general ones (e.g.
   ``NORPSI`` before ``PSI``).
2. *Modifier patterns* (``modifiers`` list, additive, substring):
   every modifier whose substring appears in the same filename is
   appended as ``" + <label>"`` to the base. This handles combination
   treatments like ``NORBAEO_HARMIN`` → ``Norbaeocystin + harmine``
   without enumerating every BASE+MODIFIER pair in the config.
3. *Qualifier regexes* (``concentrations`` / ``time_points`` /
   ``washouts`` lists, first-match-wins per list, regex): each list
   is a sequence of ``[regex, label]`` pairs. The first regex whose
   ``.search()`` hits the UPPER-cased filename contributes one
   qualifier, appended as ``" | <label>"``. Final label format is::

       <base>[ + <modifier>(s)][ | <concentration>][ | <time>][ | <washout>]

A modifier whose pattern equals the base pattern that just matched is
skipped, so a filename containing only ``HARMIN`` stays ``Harmine``
(not ``Harmine + harmine``).
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


def _load_pairs(data: dict, key: str, path) -> list[tuple[str, str]]:
    raw = data.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected '{key}' to be a list of [pattern, label] pairs"
        )
    out: list[tuple[str, str]] = []
    for entry in raw:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ValueError(
                f"{path}: each entry in '{key}' must be a [pattern, label] pair, "
                f"got {entry!r}"
            )
        pat, label = entry
        out.append((str(pat).upper(), str(label)))
    return out


def load_group_patterns(path) -> list[tuple[str, str]]:
    """Load the base ``[(pattern, group_label), ...]`` list from JSON.

    Schema::

        {
            "patterns":  [["PATTERN_UPPER", "Display group name"], ...],
            "modifiers": [["MOD_UPPER",     "modifier label"],     ...]
        }

    Only the ``"patterns"`` list is returned here (bases). Modifiers are
    loaded separately via :func:`load_group_modifiers` so existing
    callers that just want the base list keep working unchanged.

    First-match-wins on the base list; list order is significant —
    place more specific patterns before more general ones (e.g.
    ``NORPSI`` before ``PSI``). Patterns are upper-cased for
    case-insensitive matching.
    """
    import json
    from pathlib import Path
    with open(Path(path)) as f:
        data = json.load(f)
    if "patterns" not in data:
        raise ValueError(
            f"{path}: expected a top-level 'patterns' list of [pattern, label] pairs"
        )
    return _load_pairs(data, "patterns", path)


def load_group_modifiers(path) -> list[tuple[str, str]]:
    """Load the ``[(modifier_pattern, modifier_label), ...]`` list from JSON.

    Returns an empty list when the config has no ``"modifiers"`` key,
    so old configs without combination-modifier rules still work.
    """
    import json
    from pathlib import Path
    with open(Path(path)) as f:
        data = json.load(f)
    return _load_pairs(data, "modifiers", path)


def _load_regex_rules(path, key: str) -> list[tuple[re.Pattern[str], str]]:
    """Load ``[(compiled_regex, label), ...]`` from a JSON config section.

    Each entry must be a ``[regex_pattern, label]`` pair. Regexes are
    compiled with no flags (callers uppercase the input before
    searching, mirroring the substring-match stages). Returns ``[]``
    when the section is missing so old configs keep working.
    """
    import json
    from pathlib import Path
    with open(Path(path)) as f:
        data = json.load(f)
    raw = data.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected '{key}' to be a list of [regex, label] pairs"
        )
    out: list[tuple[re.Pattern[str], str]] = []
    for entry in raw:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ValueError(
                f"{path}: each entry in '{key}' must be a [regex, label] pair, "
                f"got {entry!r}"
            )
        pat, label = entry
        try:
            compiled = re.compile(str(pat))
        except re.error as exc:
            raise ValueError(
                f"{path}: invalid regex {pat!r} in '{key}': {exc}"
            ) from exc
        out.append((compiled, str(label)))
    return out


def load_concentration_patterns(path) -> list[tuple[re.Pattern[str], str]]:
    """Load the concentration regex rules from the JSON config.

    Each rule is ``[regex, label]`` (e.g.
    ``["(?<![0-9])10UM(?![A-Z0-9])", "10 uM"]``). First-match-wins.
    Returns ``[]`` when the section is absent.
    """
    return _load_regex_rules(path, "concentrations")


def load_time_patterns(path) -> list[tuple[re.Pattern[str], str]]:
    """Load the time-point regex rules from the JSON config.

    Each rule is ``[regex, label]`` (e.g.
    ``["(?<![0-9])24[_ ]?HOD(?=[^A-Z]|$)", "24h"]``). First-match-wins.
    Returns ``[]`` when the section is absent.
    """
    return _load_regex_rules(path, "time_points")


def load_washout_patterns(path) -> list[tuple[re.Pattern[str], str]]:
    """Load the washout / pulse-protocol regex rules from the JSON config.

    Each rule is ``[regex, label]`` (e.g.
    ``["1H10UM[_ ]?WO", "1h pulse"]``). First-match-wins.
    Returns ``[]`` when the section is absent.
    """
    return _load_regex_rules(path, "washouts")


def load_group_rules(path) -> dict:
    """Convenience: load every section in one call.

    Returns a dict with keys ``patterns``, ``modifiers``,
    ``concentrations``, ``time_points``, ``washouts`` — directly
    spreadable as kwargs into :func:`classify_image_by_patterns` and
    :func:`build_group_map_from_patterns`.
    """
    return {
        "patterns":       load_group_patterns(path),
        "modifiers":      load_group_modifiers(path),
        "concentrations": load_concentration_patterns(path),
        "time_points":    load_time_patterns(path),
        "washouts":       load_washout_patterns(path),
    }


def classify_image_by_patterns(
    name: str,
    patterns: Sequence[tuple[str, str]],
    modifiers: Sequence[tuple[str, str]] | None = None,
    concentrations: Sequence[tuple[re.Pattern[str], str]] | None = None,
    time_points: Sequence[tuple[re.Pattern[str], str]] | None = None,
    washouts: Sequence[tuple[re.Pattern[str], str]] | None = None,
) -> str | None:
    """Classify ``name`` into a treatment group.

    Stage 1 — pick the first base ``pattern`` (in order) that appears
    in ``name.upper()``; its label becomes the base.

    Stage 2 — if ``modifiers`` is provided, scan every modifier pattern
    over the same upper-cased filename. Each modifier found (other
    than the one whose pattern equals the base that matched) is
    appended once as ``" + <label>"``, preserving config order and
    de-duplicated by label.

    Stage 3 — for each of ``concentrations``, ``time_points``,
    ``washouts`` (in that order), the first regex whose
    ``.search(name.upper())`` succeeds contributes a single qualifier
    appended as ``" | <label>"``.

    Returns ``None`` if no base pattern matches.
    """
    upper = str(name).upper()
    base_label: str | None = None
    base_pattern: str | None = None
    for pattern, group in patterns:
        if pattern in upper:
            base_label = group
            base_pattern = pattern
            break
    if base_label is None:
        return None

    label = base_label
    if modifiers:
        seen_labels: set[str] = set()
        addons: list[str] = []
        for mpat, mlabel in modifiers:
            if mpat == base_pattern:
                continue
            if mpat in upper and mlabel not in seen_labels:
                seen_labels.add(mlabel)
                addons.append(mlabel)
        if addons:
            label = label + " + " + " + ".join(addons)

    qualifiers: list[str] = []
    for rules in (concentrations, time_points, washouts):
        if not rules:
            continue
        for rx, lbl in rules:
            if rx.search(upper):
                qualifiers.append(lbl)
                break
    if qualifiers:
        label = label + " | " + " | ".join(qualifiers)

    return label


def build_group_map_from_patterns(
    source_images: Iterable[str],
    patterns: Sequence[tuple[str, str]],
    modifiers: Sequence[tuple[str, str]] | None = None,
    concentrations: Sequence[tuple[re.Pattern[str], str]] | None = None,
    time_points: Sequence[tuple[re.Pattern[str], str]] | None = None,
    washouts: Sequence[tuple[re.Pattern[str], str]] | None = None,
) -> dict[str, str]:
    """Build ``{source_image -> group_label}`` from a pattern list.

    Optional ``modifiers`` / ``concentrations`` / ``time_points`` /
    ``washouts`` are forwarded to :func:`classify_image_by_patterns`
    so combination treatments and protocol qualifiers (concentration,
    time point, 1h pulse + washout) are encoded into the label.
    """
    gm: dict[str, str] = {}
    for img in {str(s) for s in source_images}:
        g = classify_image_by_patterns(
            img, patterns,
            modifiers=modifiers,
            concentrations=concentrations,
            time_points=time_points,
            washouts=washouts,
        )
        if g is not None:
            gm[img] = g
    return gm
