"""
Submission formatter for CLPsych 2026 Task 1.

Converts prediction dicts → task1_pred.json format as specified in guidelines.
"""

import json
from pathlib import Path


def _build_state_block(state_pred: dict) -> dict | None:
    """
    Build a single state dict for the submission JSON.
    Returns None if no elements present (presence == 1).
    """
    presence = state_pred["presence"]
    elements = state_pred["elements"]

    if not elements and presence <= 1:
        return None

    block = {"Presence": presence}
    for elem, sub_id in elements.items():
        block[elem] = {"subelement": sub_id}

    return block


def predictions_to_submission(results: list[dict]) -> list[dict]:
    """
    Convert list of prediction result dicts into the submission list format.

    Each result dict should have:
      - timeline_id, post_id
      - prediction: {adaptive: {presence, elements}, maladaptive: {presence, elements}}
    """
    submission = []
    for r in results:
        entry = {
            "timeline_id": r["timeline_id"],
            "post_id": r["post_id"],
        }

        pred = r["prediction"]

        adaptive_block = _build_state_block(pred["adaptive"])
        maladaptive_block = _build_state_block(pred["maladaptive"])

        if adaptive_block:
            entry["adaptive-state"] = adaptive_block
        if maladaptive_block:
            entry["maladaptive-state"] = maladaptive_block

        # Validator requires at least one state per entry
        if "adaptive-state" not in entry and "maladaptive-state" not in entry:
            entry["adaptive-state"] = {"Presence": 1}

        submission.append(entry)

    return submission


def save_submission(results: list[dict], output_path: str | Path) -> None:
    """Save predictions to task1_pred.json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = predictions_to_submission(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print(f"Saved {len(submission)} predictions → {output_path}")
