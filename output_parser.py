"""
Output parser for CLPsych 2026 Task 1.

Parses raw LLM text into structured prediction dicts.
Designed to be robust to minor formatting deviations from the 1B model.
"""

import re
from taxonomy import SUBELEMENTS, ELEMENTS


# ── Regex patterns ─────────────────────────────────────────────────────────────

# e.g.  ADAPTIVE_ELEMENTS: A:5, B-O:1, D:3
_ELEM_LINE_RE = re.compile(
    r"(ADAPTIVE|MALADAPTIVE)_ELEMENTS\s*:\s*([^\n]+)", re.IGNORECASE
)
# e.g.  ADAPTIVE_PRESENCE: 4
_PRES_LINE_RE = re.compile(
    r"(ADAPTIVE|MALADAPTIVE)_PRESENCE\s*:\s*(\d)", re.IGNORECASE
)
# e.g.  A:5   or   B-O:1
_SINGLE_ELEM_RE = re.compile(r"(A|B-O|B-S|C-O|C-S|D)\s*:\s*(\d+)")


def _valence_for_subelement(element: str, subelement_id: int) -> str | None:
    """Return 'adaptive' or 'maladaptive' based on which side the subelement belongs to."""
    for valence in ("adaptive", "maladaptive"):
        if subelement_id in SUBELEMENTS[element][valence]:
            return valence
    return None


def _parse_elements_line(line: str, expected_valence: str) -> dict[str, int]:
    """
    Parse an element string like 'A:5, B-O:1, D:3' into {element: subelement_id}.
    Validates that subelement_id belongs to expected_valence; skips invalid ones.
    """
    result = {}
    if re.search(r"\bNONE\b", line, re.IGNORECASE):
        return result

    for m in _SINGLE_ELEM_RE.finditer(line):
        elem = m.group(1)
        sub_id = int(m.group(2))
        actual_valence = _valence_for_subelement(elem, sub_id)
        if actual_valence == expected_valence:
            result[elem] = sub_id
        else:
            # Try to recover: find closest valid subelement index for this valence
            valid = list(SUBELEMENTS[elem][expected_valence].keys())
            if valid:
                # pick nearest
                closest = min(valid, key=lambda x: abs(x - sub_id))
                result[elem] = closest
    return result


def _clamp_presence(val: int) -> int:
    return max(1, min(5, val))


def parse_llm_output(raw_text: str) -> dict:
    """
    Parse raw LLM output into a structured prediction dict:
    {
        "adaptive":   {"presence": int, "elements": {elem: sub_id, ...}},
        "maladaptive":{"presence": int, "elements": {elem: sub_id, ...}},
    }
    """
    prediction = {
        "adaptive":    {"presence": 1, "elements": {}},
        "maladaptive": {"presence": 1, "elements": {}},
    }

    # ── Parse presence scores ──────────────────────────────────────────────────
    for m in _PRES_LINE_RE.finditer(raw_text):
        valence = m.group(1).lower()
        score = _clamp_presence(int(m.group(2)))
        prediction[valence]["presence"] = score

    # ── Parse element lines ────────────────────────────────────────────────────
    for m in _ELEM_LINE_RE.finditer(raw_text):
        valence = m.group(1).lower()
        line_content = m.group(2)
        prediction[valence]["elements"] = _parse_elements_line(line_content, valence)

    # ── Consistency fix: if no elements found, presence → 1 ───────────────────
    for valence in ("adaptive", "maladaptive"):
        if not prediction[valence]["elements"]:
            prediction[valence]["presence"] = 1

    return prediction
