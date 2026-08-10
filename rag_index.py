"""
RAG retrieval for CLPsych 2026 Task 1.

Embeds training posts with sentence-transformers, retrieves k most similar
posts with subelement-aware diversity re-ranking.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import Post
from evaluate import _gold_from_post
from taxonomy import SUBELEMENTS, ELEMENTS


# ── Embedding ────────────────────────────────────────────────────────────────

_model_cache: dict[str, SentenceTransformer] = {}

def _get_model(model_name: str = "Alibaba-NLP/gte-multilingual-base") -> SentenceTransformer:
    if model_name not in _model_cache:
        # trust_remote_code=True is required for Alibaba-NLP/gte-multilingual-base
        _model_cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _model_cache[model_name]

def _embed_texts(texts: list[str], model_name: str = "Alibaba-NLP/gte-multilingual-base") -> np.ndarray:
    model = _get_model(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

# ── Codebook (label-aware similarity) ────────────────────────────────────────

def _build_codebook(model_name: str = "Alibaba-NLP/gte-multilingual-base"):
    """Build codebook embeddings for element×valence label matching."""
    texts = []
    labels = []
    for elem in ELEMENTS:
        for valence in ("adaptive", "maladaptive"):
            desc = " ".join(SUBELEMENTS[elem][valence].values())
            if desc:
                texts.append(f"{valence} {elem}: {desc}")
                labels.append(f"{valence}_{elem}")
    return _embed_texts(texts, model_name), labels


# ── Diversity re-ranking ─────────────────────────────────────────────────────

def _element_set(gold: dict) -> set[str]:
    result = set()
    for valence in ("adaptive", "maladaptive"):
        for elem in gold[valence]["elements"]:
            result.add(f"{valence}_{elem}")
    return result


def _subelement_set(gold: dict) -> set[str]:
    """Extract set of 'valence_element:subelement' strings from gold labels."""
    result = set()
    for valence in ("adaptive", "maladaptive"):
        for elem, sub_id in gold[valence]["elements"].items():
            result.add(f"{valence}_{elem}:{sub_id}")
    return result


_TOTAL_SUBELEMENTS = sum(
    len(SUBELEMENTS[elem][valence])
    for elem in ELEMENTS
    for valence in ("adaptive", "maladaptive")
)


def _select_diverse(candidates: list[dict], k: int, alpha: float = 0.6) -> list[dict]:
    """Greedy selection maximizing alpha * similarity + (1 - alpha) * subelement coverage."""
    selected: list[dict] = []
    covered: set[str] = set()
    used: set[int] = set()

    for _ in range(k):
        best_score = -1.0
        best_idx = -1
        for i, cand in enumerate(candidates):
            if i in used:
                continue
            new_subs = _subelement_set(cand["gold"]) - covered
            coverage_gain = len(new_subs) / _TOTAL_SUBELEMENTS
            score = alpha * cand["similarity"] + (1 - alpha) * coverage_gain
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            break
        selected.append(candidates[best_idx])
        used.add(best_idx)
        covered |= _subelement_set(candidates[best_idx]["gold"])

    return selected


# ── RAGIndex ─────────────────────────────────────────────────────────────────

class RAGIndex:
    """Pre-computed embedding index for efficient repeated retrieval."""

    def __init__(self, posts: list[Post], model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        self.posts = [p for p in posts if p.has_evidence]
        self.golds = [_gold_from_post(p) for p in self.posts]
        self.timeline_ids = [p.timeline_id for p in self.posts]
        self.model_name = model_name
        self.embeddings = _embed_texts([p.text for p in self.posts], model_name)
        self._codebook_emb, self._codebook_labels = _build_codebook(model_name)

    def _get_candidates(self, query_text: str, exclude_timeline: str | None,
                        candidate_pool: int, use_codebook: bool = True) -> list[dict]:
        """Get top candidates. Uses label matching for unified, text-only for per-element."""
        query_emb = _embed_texts([query_text], self.model_name)
        text_sim = (query_emb @ self.embeddings.T)[0].copy()

        if use_codebook:
            query_labels = (query_emb @ self._codebook_emb.T)[0]
            label_scores = np.zeros(len(self.posts))
            for i, gold in enumerate(self.golds):
                gold_elems = _element_set(gold)
                for j, label in enumerate(self._codebook_labels):
                    if label in gold_elems:
                        label_scores[i] += query_labels[j]
            if label_scores.max() > 0:
                label_scores /= label_scores.max()
            scores = 0.5 * text_sim + 0.5 * label_scores
        else:
            scores = text_sim

        if exclude_timeline:
            for i, tid in enumerate(self.timeline_ids):
                if tid == exclude_timeline:
                    scores[i] = -1.0

        pool = min(candidate_pool, len(self.posts))
        top_indices = np.argsort(scores)[::-1][:pool]
        top_indices = [i for i in top_indices if scores[i] > -1.0]

        return [
            {"post": self.posts[i], "gold": self.golds[i],
             "similarity": float(text_sim[i])}
            for i in top_indices
        ]

    def retrieve(self, query_text: str, k: int = 5,
                 exclude_timeline: str | None = None,
                 candidate_pool: int = 20, diversity_alpha: float = 0.6) -> list[dict]:
        """Retrieve k diverse posts with codebook + subelement-aware re-ranking."""
        candidates = self._get_candidates(query_text, exclude_timeline, candidate_pool, use_codebook=True)
        return _select_diverse(candidates, k, diversity_alpha)

    def retrieve_for_element(self, query_text: str, target_element: str,
                             k: int = 5, exclude_timeline: str | None = None,
                             candidate_pool: int = 30) -> list[dict]:
        """Retrieve posts maximizing subelement diversity for one element (text-only candidates)."""
        candidates = self._get_candidates(query_text, exclude_timeline, candidate_pool, use_codebook=False)

        all_sub_ids = set()
        for valence in ("adaptive", "maladaptive"):
            for sub_id in SUBELEMENTS[target_element][valence]:
                all_sub_ids.add(f"{valence}_{sub_id}")
        max_coverage = len(all_sub_ids) or 1

        selected = []
        covered = set()

        for _ in range(k):
            best_score = -1
            best_idx = -1
            for i, cand in enumerate(candidates):
                if cand in selected:
                    continue
                cand_subs = set()
                for valence in ("adaptive", "maladaptive"):
                    elems = cand["gold"][valence]["elements"]
                    if target_element in elems:
                        cand_subs.add(f"{valence}_{elems[target_element]}")
                new_subs = cand_subs - covered
                score = 0.4 * cand["similarity"] + 0.6 * (len(new_subs) / max_coverage)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(candidates[best_idx])
            for valence in ("adaptive", "maladaptive"):
                elems = candidates[best_idx]["gold"][valence]["elements"]
                if target_element in elems:
                    covered.add(f"{valence}_{elems[target_element]}")

        return selected
