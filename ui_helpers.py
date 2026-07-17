from __future__ import annotations


def format_score(value: float, *, scale: str = "ratio", decimals: int = 1) -> str:
    """Return a human-readable score with its denominator."""
    if scale == "ten":
        return f"{round(float(value) * 10, decimals)}/{10}"
    if scale == "twenty":
        return f"{round(float(value), decimals)}/{20}"
    if scale == "percent":
        return f"{round(float(value) * 100, decimals)}%"
    return f"{round(float(value), decimals)}/{1}"


def format_score_with_context(value: float, *, scale: str = "ratio", decimals: int = 1) -> str:
    """Return a concise label suitable for UI cards."""
    if scale == "ten":
        return f"{round(float(value) * 10, decimals)} / 10"
    if scale == "twenty":
        return f"{round(float(value), decimals)} / 20"
    if scale == "percent":
        return f"{round(float(value) * 100, decimals)} %"
    return f"{round(float(value), decimals)} / 1"


def candidates_to_analyze(candidates: list[dict], job_id: str) -> list[dict]:
    """Return candidates that still need analysis or need a retry after an error."""
    pending: list[dict] = []
    for candidate in candidates:
        result = candidate.get("results", {}).get(job_id)
        if result is None or result.get("status") == "error":
            pending.append(candidate)
    return pending
