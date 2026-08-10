"""Tests for CLPsych 2026 pipeline."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonomy import ELEMENTS, SUBELEMENTS
from output_parser import parse_llm_output
from prompt_builder_rag import (
    build_messages, build_messages_per_element,
    _truncate, _format_gold_as_lines, _extract_evidence, MAX_CHARS,
)
from evaluate import _compute_subelement_macro_f1, _gold_from_post


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_gold(adaptive_elems=None, maladaptive_elems=None,
               adaptive_presence=1, maladaptive_presence=1):
    return {
        "adaptive": {"presence": adaptive_presence, "elements": adaptive_elems or {}},
        "maladaptive": {"presence": maladaptive_presence, "elements": maladaptive_elems or {}},
    }


class FakeState:
    def __init__(self):
        self.presence = None
        self.A = self.B_O = self.B_S = self.C_O = self.C_S = self.D = None


class FakeEvidence:
    def __init__(self, category, highlighted_evidence):
        self.category = category
        self.highlighted_evidence = highlighted_evidence


class FakePost:
    def __init__(self, text="test", timeline_id="t1", post_id="p1"):
        self.text = text
        self.timeline_id = timeline_id
        self.post_id = post_id
        self.adaptive_state = FakeState()
        self.maladaptive_state = FakeState()


def _make_rag_result(text, gold, similarity=0.8):
    return {"post": FakePost(text), "gold": gold, "similarity": similarity}


# ── output_parser ────────────────────────────────────────────────────────────

class TestOutputParser:
    def test_parse_full_output(self):
        raw = """ADAPTIVE_ELEMENTS: A:5, B-O:1, D:3
ADAPTIVE_PRESENCE: 4
MALADAPTIVE_ELEMENTS: A:4, B-S:2
MALADAPTIVE_PRESENCE: 3"""
        pred = parse_llm_output(raw)
        assert pred["adaptive"]["elements"] == {"A": 5, "B-O": 1, "D": 3}
        assert pred["adaptive"]["presence"] == 4
        assert pred["maladaptive"]["elements"] == {"A": 4, "B-S": 2}
        assert pred["maladaptive"]["presence"] == 3

    def test_parse_none_elements(self):
        raw = """ADAPTIVE_ELEMENTS: NONE
ADAPTIVE_PRESENCE: 1
MALADAPTIVE_ELEMENTS: NONE
MALADAPTIVE_PRESENCE: 1"""
        pred = parse_llm_output(raw)
        assert pred["adaptive"]["elements"] == {}
        assert pred["adaptive"]["presence"] == 1

    def test_empty_input(self):
        pred = parse_llm_output("")
        assert pred["adaptive"]["elements"] == {}
        assert pred["adaptive"]["presence"] == 1

    def test_presence_clamped(self):
        raw = "ADAPTIVE_ELEMENTS: A:5\nADAPTIVE_PRESENCE: 9\nMALADAPTIVE_PRESENCE: 0"
        pred = parse_llm_output(raw)
        assert pred["adaptive"]["presence"] == 5  # clamped from 9
        assert pred["maladaptive"]["presence"] == 1  # no elements → consistency fix

    def test_roundtrip(self):
        gold = _make_gold(
            adaptive_elems={"A": 5, "C-S": 1},
            adaptive_presence=3,
            maladaptive_elems={"A": 4, "D": 6},
            maladaptive_presence=4,
        )
        lines = _format_gold_as_lines(gold)
        parsed = parse_llm_output(lines)
        assert parsed["adaptive"]["elements"] == gold["adaptive"]["elements"]
        assert parsed["adaptive"]["presence"] == gold["adaptive"]["presence"]
        assert parsed["maladaptive"]["elements"] == gold["maladaptive"]["elements"]
        assert parsed["maladaptive"]["presence"] == gold["maladaptive"]["presence"]


# ── prompt_builder_rag ───────────────────────────────────────────────────────

class TestPromptBuilder:
    def test_build_messages_structure(self):
        gold = _make_gold(adaptive_elems={"A": 5})
        results = [_make_rag_result("example", gold)]
        msgs = build_messages("query", results)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_taxonomy_in_system(self):
        msgs = build_messages("test", [])
        system = msgs[0]["content"]
        for elem in ELEMENTS:
            assert f"[{elem}]" in system

    def test_taxonomy_full_words(self):
        msgs = build_messages("test", [])
        system = msgs[0]["content"]
        assert "ADAPTIVE 1:" in system
        assert "MALADAPTIVE 2:" in system

    def test_presence_scale_in_system(self):
        msgs = build_messages("test", [])
        assert "1 = Not present" in msgs[0]["content"]

    def test_examples_in_user(self):
        gold = _make_gold(adaptive_elems={"A": 5})
        results = [_make_rag_result("example text here", gold)]
        msgs = build_messages("query", results)
        assert "example text here" in msgs[1]["content"]
        assert "example text here" not in msgs[0]["content"]

    def test_truncation(self):
        long = "a" * (MAX_CHARS + 200)
        t = _truncate(long)
        assert len(t) == MAX_CHARS + 3
        assert t.endswith("...")

    def test_no_truncation_short(self):
        short = "hello"
        assert _truncate(short) == "hello"

    def test_gold_element_order(self):
        gold = _make_gold(adaptive_elems={"D": 1, "A": 5, "B-S": 1})
        lines = _format_gold_as_lines(gold)
        assert lines.index("A:5") < lines.index("B-S:1") < lines.index("D:1")

    def test_per_element_no_presence(self):
        gold = _make_gold(adaptive_elems={"A": 5})
        results = [_make_rag_result("text", gold)]
        msgs = build_messages_per_element("query", "A", results)
        system = msgs[0]["content"]
        assert "PRESENCE" not in system
        assert "ONLY evaluate the A element" in system

    def test_evidence_bound(self):
        post = FakePost("test")
        post.adaptive_state.A = FakeEvidence("(5) Content", "feeling happy")
        post.maladaptive_state.D = FakeEvidence("(6) Competence", "I give up")
        gold = _make_gold(adaptive_elems={"A": 5}, maladaptive_elems={"D": 6})
        evidence = _extract_evidence(post, gold)
        assert len(evidence) == 2
        assert evidence[0] == ("ADAPTIVE A:5", '"feeling happy"')
        assert evidence[1] == ("MALADAPTIVE D:6", '"I give up"')

    def test_evidence_target_element(self):
        post = FakePost("test")
        post.adaptive_state.A = FakeEvidence("(5) Content", "happy")
        post.adaptive_state.D = FakeEvidence("(1) Relatedness", "connect")
        gold = _make_gold(adaptive_elems={"A": 5, "D": 1})
        evidence = _extract_evidence(post, gold, target_element="A")
        assert len(evidence) == 1
        assert evidence[0][0] == "ADAPTIVE A:5"


# ── evaluate ─────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_subelement_macro_f1_correct_predictions(self):
        golds = [_make_gold(adaptive_elems={"A": 5}), _make_gold(adaptive_elems={"A": 3})]
        preds = [_make_gold(adaptive_elems={"A": 5}), _make_gold(adaptive_elems={"A": 3})]
        result = _compute_subelement_macro_f1(golds, preds, "adaptive", "A")
        # Classes 5 and 3 have perfect F1=1.0, other 5 classes have F1=0.0
        # Macro F1 = 2/7 ≈ 0.286
        assert result["per_class"][5]["f1"] == 1.0
        assert result["per_class"][3]["f1"] == 1.0
        assert result["macro_f1"] == pytest.approx(2 / 7, abs=0.01)

    def test_subelement_macro_f1_wrong_subelement(self):
        golds = [_make_gold(adaptive_elems={"A": 5})]
        preds = [_make_gold(adaptive_elems={"A": 3})]
        result = _compute_subelement_macro_f1(golds, preds, "adaptive", "A")
        assert result["macro_f1"] == 0.0  # wrong class for both 5 and 3

    def test_subelement_macro_f1_absent_no_penalty(self):
        golds = [_make_gold(), _make_gold()]
        preds = [_make_gold(), _make_gold()]
        result = _compute_subelement_macro_f1(golds, preds, "adaptive", "A")
        assert result["macro_f1"] == 0.0  # no positive classes to evaluate

    def test_subelement_false_positive_penalty(self):
        golds = [_make_gold()]  # A absent
        preds = [_make_gold(adaptive_elems={"A": 5})]  # A:5 predicted
        result = _compute_subelement_macro_f1(golds, preds, "adaptive", "A")
        # FP for class 5, so F1 for class 5 < 1.0
        assert result["per_class"][5]["precision"] == 0.0


# ── rag_index ────────────────────────────────────────────────────────────────

class TestRAGIndex:
    def test_subelement_set(self):
        from rag_index import _subelement_set
        gold = _make_gold(adaptive_elems={"A": 5, "D": 1}, maladaptive_elems={"A": 4})
        subs = _subelement_set(gold)
        assert subs == {"adaptive_A:5", "adaptive_D:1", "maladaptive_A:4"}

    def test_select_diverse_covers_subelements(self):
        from rag_index import _select_diverse
        candidates = [
            {"gold": _make_gold(adaptive_elems={"A": 5}), "similarity": 0.9},
            {"gold": _make_gold(adaptive_elems={"A": 5}), "similarity": 0.8},
            {"gold": _make_gold(adaptive_elems={"A": 3}, maladaptive_elems={"D": 6}), "similarity": 0.5},
        ]
        selected = _select_diverse(candidates, k=2, alpha=0.0)
        # With alpha=0, should pick for max coverage
        subs = set()
        for s in selected:
            from rag_index import _subelement_set
            subs |= _subelement_set(s["gold"])
        assert len(subs) >= 3
