"""
Data loader for CLPsych 2026 Task 1.
Reads the 30 training JSON files from train_tasks12/.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceEntry:
    category: str
    highlighted_evidence: str


@dataclass
class SelfState:
    presence: Optional[int] = None
    A: Optional[EvidenceEntry] = None
    B_O: Optional[EvidenceEntry] = None
    B_S: Optional[EvidenceEntry] = None
    C_O: Optional[EvidenceEntry] = None
    C_S: Optional[EvidenceEntry] = None
    D: Optional[EvidenceEntry] = None


@dataclass
class Post:
    timeline_id: str
    post_id: str
    post_index: int
    date: str
    text: str
    well_being: Optional[float]
    switch_label: str           # "S" or "0"
    escalation_label: str       # "E" or "0"
    adaptive_state: SelfState = field(default_factory=SelfState)
    maladaptive_state: SelfState = field(default_factory=SelfState)
    has_evidence: bool = False   # True if annotated evidence exists


def _parse_element(raw: dict) -> Optional[EvidenceEntry]:
    if not raw:
        return None
    return EvidenceEntry(
        category=raw.get("Category", ""),
        highlighted_evidence=raw.get("highlighted_evidence", ""),
    )


def _parse_state(raw: dict, valence: str) -> SelfState:
    if not raw:
        return SelfState()
    presence = raw.get("Presence")
    return SelfState(
        presence=int(presence) if presence is not None else None,
        A=_parse_element(raw.get("A")),
        B_O=_parse_element(raw.get("B-O")),
        B_S=_parse_element(raw.get("B-S")),
        C_O=_parse_element(raw.get("C-O")),
        C_S=_parse_element(raw.get("C-S")),
        D=_parse_element(raw.get("D")),
    )


def _has_evidence(post_raw: dict) -> bool:
    """A post has evidence if either state has at least one ABCD element annotated."""
    for valence in ("adaptive-state", "maladaptive-state"):
        state = post_raw.get("evidence", {}).get(valence, {})
        for key in ["A", "B-O", "B-S", "C-O", "C-S", "D"]:
            if key in state:
                return True
    return False


def load_timeline(filepath: str | Path) -> list[Post]:
    """Load a single timeline JSON file and return a list of Post objects."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    timeline_id = data["timeline_id"]
    posts = []

    for raw in data["posts"]:
        evidence = raw.get("evidence", {})
        post = Post(
            timeline_id=timeline_id,
            post_id=raw["post_id"],
            post_index=raw["post_index"],
            date=raw.get("date", ""),
            text=raw.get("post", ""),
            well_being=raw.get("Well-being"),
            switch_label=str(raw.get("Switch", "0")),
            escalation_label=str(raw.get("Escalation", "0")),
            adaptive_state=_parse_state(evidence.get("adaptive-state", {}), "adaptive"),
            maladaptive_state=_parse_state(evidence.get("maladaptive-state", {}), "maladaptive"),
            has_evidence=_has_evidence(raw),
        )
        posts.append(post)

    return posts


def load_all_timelines(train_dir: str | Path) -> list[Post]:
    """Load all timeline JSON files from a directory."""
    train_dir = Path(train_dir)
    all_posts: list[Post] = []
    files = sorted(train_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files found in {train_dir}")

    print(f"Loading {len(files)} timeline files from {train_dir} ...")
    for fp in files:
        posts = load_timeline(fp)
        all_posts.extend(posts)
        print(f"  {fp.name}: {len(posts)} posts (timeline {posts[0].timeline_id})")

    evidenced = [p for p in all_posts if p.has_evidence]
    print(f"\nTotal posts loaded  : {len(all_posts)}")
    print(f"Posts with evidence : {len(evidenced)} ({len(evidenced)/len(all_posts)*100:.1f}%)")
    return all_posts
