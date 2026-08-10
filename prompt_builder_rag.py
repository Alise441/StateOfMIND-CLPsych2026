"""
Prompt builder for CLPsych 2026 Task 1.

Three public functions:
  build_messages()             → unified prompt (all elements)
  build_messages_per_element() → per-element prompt (one element)
  build_messages_adaptive()    → adaptive-focused prompt (with maladaptive context)
"""

from taxonomy import (
    ELEMENTS, SUBELEMENTS, ELEMENT_DEFINITIONS, VALENCE_DEFINITIONS,
    PRESENCE_SCALE as PRESENCE_SCALE_DICT,
)

MAX_CHARS = 1800

TASK_CONTEXT = """You are a clinical psychology annotation assistant trained in the MIND framework.
The MIND framework analyzes social media posts to identify psychological self-states.
Each post may express adaptive and/or maladaptive psychological states \
across 6 dimensions: Affect (A), Behavior toward Others (B-O), Behavior toward Self (B-S), \
Cognition about Others (C-O), Cognition about Self (C-S), and Desire (D).
Your job: identify which specific subelements are present and rate their overall presence."""

PRESENCE_SCALE_TEXT = "PRESENCE SCALE (reflects psychological centrality, not mere frequency of words):\n" + \
    "\n".join(f"{k} = {v}" for k, v in PRESENCE_SCALE_DICT.items())

FULL_RULES = """RULES:
1. For each ABCD element (A, B-O, B-S, C-O, C-S, D), decide if it is present.
2. If present, choose the single most prominent subelement — the one most strongly \
expressed or most central to the post's meaning.
3. Assign ADAPTIVE_PRESENCE and MALADAPTIVE_PRESENCE scores (1-5).
4. Output ONLY these 4 lines: ADAPTIVE_ELEMENTS, ADAPTIVE_PRESENCE, MALADAPTIVE_ELEMENTS, \
MALADAPTIVE_PRESENCE. No evidence lines, no extra text.
5. If no adaptive elements: write ADAPTIVE_ELEMENTS: NONE and ADAPTIVE_PRESENCE: 1
6. If no maladaptive elements: write MALADAPTIVE_ELEMENTS: NONE and MALADAPTIVE_PRESENCE: 1
7. Only annotate elements you are confident about. It is better to miss an element \
than to predict the wrong subelement."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _truncate(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS] + "..."


def _build_taxonomy_block(elements=None) -> str:
    elems = elements or ELEMENTS
    if isinstance(elems, str):
        elems = [elems]
    lines = [
        "ABCD TAXONOMY (use exact subelement numbers):",
        "",
        VALENCE_DEFINITIONS["adaptive"],
        VALENCE_DEFINITIONS["maladaptive"],
    ]
    for elem in elems:
        lines.append(f"\n  [{elem}] {ELEMENT_DEFINITIONS.get(elem, '')}")
        for valence in ("adaptive", "maladaptive"):
            tag = "ADAPTIVE" if valence == "adaptive" else "MALADAPTIVE"
            for idx, desc in SUBELEMENTS[elem][valence].items():
                lines.append(f"    {tag} {idx}: {desc}")
    return "\n".join(lines)


TAXONOMY_BLOCK = _build_taxonomy_block()


def _extract_evidence(post_obj, gold: dict, target_element: str = None) -> list[tuple[str, str]]:
    results = []
    def _add(elements_dict, state_obj, valence_tag):
        if not elements_dict or not state_obj:
            return
        for elem, sub_id in elements_dict.items():
            if target_element and elem != target_element:
                continue
            attr_name = elem.replace("-", "_")
            if hasattr(state_obj, attr_name):
                entry = getattr(state_obj, attr_name)
                if entry and entry.highlighted_evidence:
                    clean = " ".join(entry.highlighted_evidence.split())
                    results.append((f"{valence_tag} {elem}:{sub_id}", f'"{clean}"'))
    _add(gold.get("adaptive", {}).get("elements", {}), post_obj.adaptive_state, "ADAPTIVE")
    _add(gold.get("maladaptive", {}).get("elements", {}), post_obj.maladaptive_state, "MALADAPTIVE")
    return results


def _format_gold_as_lines(gold: dict) -> str:
    lines = []
    for valence in ("adaptive", "maladaptive"):
        tag = valence.upper()
        elements = gold[valence]["elements"]
        presence = gold[valence].get("presence") or 1
        if elements:
            pairs = [f"{e}:{elements[e]}" for e in ELEMENTS if e in elements]
            lines.append(f"{tag}_ELEMENTS: {', '.join(pairs)}")
        else:
            lines.append(f"{tag}_ELEMENTS: NONE")
        lines.append(f"{tag}_PRESENCE: {presence}")
    return "\n".join(lines)


def _format_example(post_obj, gold: dict) -> str:
    evidence = _extract_evidence(post_obj, gold)
    evidence_block = "\n".join(f"{label} EVIDENCE: {text}" for label, text in evidence)
    if evidence_block:
        evidence_block += "\n"
    return (
        f"EXAMPLE\n-------\n"
        f"POST: \"{_truncate(post_obj.text)}\"\n"
        f"{evidence_block}"
        f"{_format_gold_as_lines(gold)}\n-------"
    )


def _build_user_content(rag_results: list[dict], query_text: str,
                        format_fn=None) -> str:
    format_fn = format_fn or (lambda r: _format_example(r["post"], r["gold"]))
    parts = []
    if rag_results:
        examples = "\n\n".join(format_fn(r) for r in rag_results)
        parts.append(f"Here are {len(rag_results)} annotated examples:\n")
        parts.append(examples)
        parts.append(f"\nNow annotate the following post:")
    parts.append(f'POST: "{_truncate(query_text)}"')
    return "\n".join(parts)


# ── Public functions ─────────────────────────────────────────────────────────

def build_messages(query_text: str, rag_results: list[dict]) -> list[dict]:
    """Unified prompt: all elements + presence."""
    system = f"{TASK_CONTEXT}\n\n{TAXONOMY_BLOCK}\n\n{PRESENCE_SCALE_TEXT}\n\n{FULL_RULES}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(rag_results, query_text)},
    ]


def build_messages_per_element(query_text: str, target_element: str,
                               rag_results: list[dict]) -> list[dict]:
    """Per-element prompt: one element only, no presence."""
    taxonomy = _build_taxonomy_block(target_element)
    rules = f"""RULES:
1. ONLY evaluate the {target_element} element. Decide if it is present.
2. If present, choose the single most prominent subelement — the one most strongly \
expressed or most central to the post's meaning.
3. Output ONLY these 2 lines: ADAPTIVE_ELEMENTS and MALADAPTIVE_ELEMENTS. No presence scores.
4. If adaptive element {target_element} is not present: write ADAPTIVE_ELEMENTS: NONE
5. If maladaptive element {target_element} is not present: write MALADAPTIVE_ELEMENTS: NONE
6. Only annotate if you are confident. It is better to predict NONE than the wrong subelement."""

    system = f"{TASK_CONTEXT}\n\n{taxonomy}\n\n{rules}"

    def format_per_element(r):
        post, gold = r["post"], r["gold"]
        adapt = f"{target_element}:{gold['adaptive']['elements'][target_element]}" \
            if target_element in gold["adaptive"]["elements"] else "NONE"
        malad = f"{target_element}:{gold['maladaptive']['elements'][target_element]}" \
            if target_element in gold["maladaptive"]["elements"] else "NONE"
        evidence = _extract_evidence(post, gold, target_element)
        ev_block = "\n".join(f"{l} EVIDENCE: {t}" for l, t in evidence)
        if ev_block:
            ev_block += "\n"
        return (
            f"EXAMPLE\n-------\n"
            f"POST: \"{_truncate(post.text)}\"\n"
            f"{ev_block}"
            f"ADAPTIVE_ELEMENTS: {adapt}\n"
            f"MALADAPTIVE_ELEMENTS: {malad}\n-------"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(rag_results, query_text, format_per_element)},
    ]


def _build_adaptive_taxonomy_block() -> str:
    """Build taxonomy block with only adaptive subelements."""
    lines = [
        "ADAPTIVE TAXONOMY (use exact subelement numbers):",
        "",
        VALENCE_DEFINITIONS["adaptive"],
    ]
    for elem in ELEMENTS:
        lines.append(f"\n  [{elem}] {ELEMENT_DEFINITIONS.get(elem, '')}")
        for idx, desc in SUBELEMENTS[elem]["adaptive"].items():
            lines.append(f"    ADAPTIVE {idx}: {desc}")
    return "\n".join(lines)


ADAPTIVE_TASK_CONTEXT = """You are a clinical psychology annotation assistant trained in the MIND framework.
Your task: identify ADAPTIVE psychological patterns that co-exist alongside \
maladaptive states in this post.
Important: Adaptive states often co-exist with maladaptive states. A person \
describing depression may also show self-care, self-acceptance, healthy grieving, \
or reaching out for support. Look carefully for these adaptive signals."""

ADAPTIVE_RULES = """RULES:
1. Focus ONLY on ADAPTIVE elements. Maladaptive annotation is already done.
2. For each element, decide if an adaptive subelement is present.
3. If present, choose the most prominent adaptive subelement.
4. Assign ADAPTIVE_PRESENCE (1-5).
5. Output ONLY these 2 lines: ADAPTIVE_ELEMENTS and ADAPTIVE_PRESENCE. No extra text.
6. If no adaptive elements: ADAPTIVE_ELEMENTS: NONE and ADAPTIVE_PRESENCE: 1
7. For B-S and C-S: predict whenever you see ANY signal — only one possible subelement.
8. For A, B-O, C-O, D: only annotate if confident in the specific subelement."""


def build_messages_adaptive(query_text: str, rag_results: list[dict],
                            maladaptive_pred: dict) -> list[dict]:
    """Adaptive-focused prompt with maladaptive context from unified call.

    Args:
        query_text: the post text
        rag_results: RAG examples (full gold, both valences)
        maladaptive_pred: {"elements": {elem: sub_id}, "presence": int} from unified
    """
    taxonomy = _build_adaptive_taxonomy_block()
    system = f"{ADAPTIVE_TASK_CONTEXT}\n\n{taxonomy}\n\n{PRESENCE_SCALE_TEXT}\n\n{ADAPTIVE_RULES}"

    # Format maladaptive context
    mal_elems = maladaptive_pred.get("elements", {})
    mal_pres = maladaptive_pred.get("presence", 1)
    if mal_elems:
        mal_pairs = ", ".join(f"{e}:{mal_elems[e]}" for e in ELEMENTS if e in mal_elems)
        mal_context = (
            f"The following maladaptive patterns were already identified:\n"
            f"MALADAPTIVE_ELEMENTS: {mal_pairs}\n"
            f"MALADAPTIVE_PRESENCE: {mal_pres}\n"
        )
    else:
        mal_context = "No maladaptive patterns were identified in this post.\n"

    # Build user content with RAG examples (full gold) + maladaptive context
    parts = []
    if rag_results:
        examples = "\n\n".join(_format_example(r["post"], r["gold"]) for r in rag_results)
        parts.append(f"Here are {len(rag_results)} annotated examples:\n")
        parts.append(examples)
        parts.append("")

    parts.append(mal_context)
    parts.append(f"Now identify ADAPTIVE elements in this post:")
    parts.append(f'POST: "{_truncate(query_text)}"')

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]
