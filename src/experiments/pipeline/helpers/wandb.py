"""Small W&B naming helpers for experiment launchers."""

from __future__ import annotations


def default_project_for_mode(
    *,
    mode: str,
    screening_project: str,
    final_project: str,
    cadence_project: str | None = None,
) -> str:
    if mode == "screening":
        return screening_project
    if mode == "cadence" and cadence_project is not None:
        return cadence_project
    return final_project


def group_name(*parts: str | None) -> str:
    return "_".join(str(part) for part in parts if part)


def tags(*parts: object) -> list[str]:
    out: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            out.extend(str(item) for item in part if item is not None and str(item))
        elif str(part):
            out.append(str(part))
    return out
