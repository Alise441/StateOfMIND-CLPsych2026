"""
Evaluation for CLPsych 2026 Task 1.

Task 1.1: ABCD Element & Subelement Classification
  - Element presence: binary F1 per element (reported, NOT ranking)
  - Subelement classification: multi-class macro F1 per element (RANKING metric)
  - Ranking: mean(Adaptive Macro F1, Maladaptive Macro F1)

Task 1.2: Presence Rating
  - MAE, RMSE, QWK, Spearman
  - Ranking: mean(Adaptive RMSE, Maladaptive RMSE) — lower is better
"""

import math
import re
from collections import defaultdict
from data_loader import Post
from taxonomy import ELEMENTS, SUBELEMENTS


# ── Helper: extract gold labels from a Post ────────────────────────────────────

def _gold_from_post(post: Post) -> dict:
    """
    Returns gold dict:
    {
      "adaptive":   {"presence": int|None, "elements": {elem: sub_id}},
      "maladaptive":{"presence": int|None, "elements": {elem: sub_id}},
    }
    """
    def state_to_dict(state):
        elements = {}
        for attr, elem_key in [
            ("A", "A"), ("B_O", "B-O"), ("B_S", "B-S"),
            ("C_O", "C-O"), ("C_S", "C-S"), ("D", "D"),
        ]:
            ev = getattr(state, attr)
            if ev is not None:
                m = re.match(r"\((\d+)\)", ev.category.strip())
                if m:
                    elements[elem_key] = int(m.group(1))
        return {"presence": state.presence, "elements": elements}

    return {
        "adaptive":    state_to_dict(post.adaptive_state),
        "maladaptive": state_to_dict(post.maladaptive_state),
    }


# ── Task 1.1: Element presence F1 (reported, NOT ranking) ────────────────────

def _compute_element_f1(
    gold_list: list[dict],
    pred_list: list[dict],
    valence: str,
    element: str,
) -> dict:
    tp = fp = fn = 0
    for gold, pred in zip(gold_list, pred_list):
        g_present = element in gold[valence]["elements"]
        p_present = element in pred[valence]["elements"]

        if g_present and p_present:
            tp += 1
        elif not g_present and p_present:
            fp += 1
        elif g_present and not p_present:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


# ── Task 1.1: Subelement multi-class macro F1 (RANKING metric) ───────────────

def _compute_subelement_macro_f1(
    gold_list: list[dict],
    pred_list: list[dict],
    valence: str,
    element: str,
) -> dict:
    """Per-class F1 for each valid subelement, then macro average.

    Classes = valid subelement IDs for this element×valence.
    "Absent" (0) is excluded from F1 labels, but errors with absent
    penalize positive classes via FP/FN.
    """
    valid_ids = list(SUBELEMENTS[element][valence].keys())

    # Collect gold/pred subelement IDs (0 = absent)
    y_true = []
    y_pred = []
    for gold, pred in zip(gold_list, pred_list):
        g_elems = gold[valence]["elements"]
        p_elems = pred[valence]["elements"]
        y_true.append(g_elems.get(element, 0))
        y_pred.append(p_elems.get(element, 0))

    # Per-class F1 (only for valid subelement IDs, not for 0=absent)
    per_class_f1 = {}
    for sub_id in valid_ids:
        tp = fp = fn = 0
        for gt, pr in zip(y_true, y_pred):
            if gt == sub_id and pr == sub_id:
                tp += 1
            elif gt != sub_id and pr == sub_id:
                fp += 1
            elif gt == sub_id and pr != sub_id:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        per_class_f1[sub_id] = {"f1": f1, "precision": precision, "recall": recall, "support": support}

    # Macro F1 = average over all valid subelement classes
    f1_values = [v["f1"] for v in per_class_f1.values()]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {"macro_f1": macro_f1, "per_class": per_class_f1}


# ── Task 1.1: Aggregate ──────────────────────────────────────────────────────

def evaluate_task11(posts: list[Post], results: list[dict]) -> dict:
    """Evaluate Task 1.1: element presence F1 + subelement macro F1."""
    evidenced_pairs = [
        (p, r) for p, r in zip(posts, results) if p.has_evidence
    ]
    if not evidenced_pairs:
        print("No evidenced posts for evaluation.")
        return {}

    posts_ev, results_ev = zip(*evidenced_pairs)
    gold_list = [_gold_from_post(p) for p in posts_ev]
    pred_list = [r["prediction"] for r in results_ev]

    metrics = {}
    for valence in ("adaptive", "maladaptive"):
        metrics[valence] = {}
        elem_f1_scores = []
        sub_macro_f1_scores = []

        for elem in ELEMENTS:
            # Element presence F1 (reported, not ranking)
            m_elem = _compute_element_f1(gold_list, pred_list, valence, elem)
            metrics[valence][elem] = m_elem
            elem_f1_scores.append(m_elem["f1"])

            # Subelement multi-class macro F1 (ranking metric)
            m_sub = _compute_subelement_macro_f1(gold_list, pred_list, valence, elem)
            metrics[valence][f"{elem}_sub"] = m_sub
            sub_macro_f1_scores.append(m_sub["macro_f1"])

        metrics[valence]["elem_macro_f1"] = sum(elem_f1_scores) / len(elem_f1_scores)
        metrics[valence]["sub_macro_f1"] = sum(sub_macro_f1_scores) / len(sub_macro_f1_scores)

    # Overall (both are averages of adaptive + maladaptive)
    metrics["overall_elem_macro_f1"] = (
        metrics["adaptive"]["elem_macro_f1"] + metrics["maladaptive"]["elem_macro_f1"]
    ) / 2
    metrics["overall_sub_macro_f1"] = (
        metrics["adaptive"]["sub_macro_f1"] + metrics["maladaptive"]["sub_macro_f1"]
    ) / 2

    return metrics


# ── Task 1.2: Presence score metrics ─────────────────────────────────────────

def _quadratic_weighted_kappa(y_true: list[int], y_pred: list[int]) -> float:
    """Compute QWK for ordinal ratings (1-5 scale)."""
    min_val, max_val = 1, 5
    num_classes = max_val - min_val + 1

    # Build confusion matrix
    conf = [[0] * num_classes for _ in range(num_classes)]
    for gt, pr in zip(y_true, y_pred):
        gt_idx = max(0, min(num_classes - 1, gt - min_val))
        pr_idx = max(0, min(num_classes - 1, pr - min_val))
        conf[gt_idx][pr_idx] += 1

    n = len(y_true)
    if n == 0:
        return 0.0

    # Weight matrix (quadratic)
    weight = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            weight[i][j] = (i - j) ** 2 / (num_classes - 1) ** 2

    # Expected matrix
    row_sums = [sum(conf[i]) for i in range(num_classes)]
    col_sums = [sum(conf[i][j] for i in range(num_classes)) for j in range(num_classes)]
    expected = [[row_sums[i] * col_sums[j] / n for j in range(num_classes)] for i in range(num_classes)]

    # QWK
    num = sum(weight[i][j] * conf[i][j] for i in range(num_classes) for j in range(num_classes))
    den = sum(weight[i][j] * expected[i][j] for i in range(num_classes) for j in range(num_classes))

    return 1.0 - num / den if den > 0 else 0.0


def _spearman_correlation(y_true: list, y_pred: list) -> float:
    """Compute Spearman rank correlation."""
    n = len(y_true)
    if n < 2:
        return 0.0

    def _rank(values):
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and values[sorted_indices[j]] == values[sorted_indices[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rank_true = _rank(y_true)
    rank_pred = _rank(y_pred)

    d_sq = sum((rt - rp) ** 2 for rt, rp in zip(rank_true, rank_pred))
    return 1.0 - (6 * d_sq) / (n * (n ** 2 - 1))


def evaluate_task12(posts: list[Post], results: list[dict]) -> dict:
    """Evaluate Task 1.2: MAE, RMSE, QWK, Spearman for presence scores."""
    evidenced_pairs = [
        (p, r) for p, r in zip(posts, results) if p.has_evidence
    ]
    if not evidenced_pairs:
        return {}

    posts_ev, results_ev = zip(*evidenced_pairs)
    gold_list = [_gold_from_post(p) for p in posts_ev]
    pred_list = [r["prediction"] for r in results_ev]

    metrics = {}
    for valence in ("adaptive", "maladaptive"):
        y_true = []
        y_pred = []
        for gold, pred in zip(gold_list, pred_list):
            g_pres = gold[valence]["presence"]
            p_pres = pred[valence]["presence"]
            if g_pres is not None:
                y_true.append(g_pres)
                y_pred.append(p_pres)

        if y_true:
            errors = [abs(g - p) for g, p in zip(y_true, y_pred)]
            mae  = sum(errors) / len(errors)
            rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
            qwk  = _quadratic_weighted_kappa(y_true, y_pred)
            spearman = _spearman_correlation(y_true, y_pred)
            metrics[valence] = {
                "mae": mae, "rmse": rmse, "qwk": qwk,
                "spearman": spearman, "n": len(y_true),
            }
        else:
            metrics[valence] = {"mae": None, "rmse": None, "qwk": None, "spearman": None, "n": 0}

    if metrics["adaptive"]["rmse"] is not None and metrics["maladaptive"]["rmse"] is not None:
        metrics["combined_rmse"] = (
            metrics["adaptive"]["rmse"] + metrics["maladaptive"]["rmse"]
        ) / 2

    return metrics


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_evaluation(task11_metrics: dict, task12_metrics: dict) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # ── Task 1.1 ──
    print("\n── Task 1.1: Element & Subelement Classification ─────────────")
    for valence in ("adaptive", "maladaptive"):
        print(f"\n  [{valence.upper()}]")
        for elem in ELEMENTS:
            m = task11_metrics.get(valence, {}).get(elem, {})
            m_sub = task11_metrics.get(valence, {}).get(f"{elem}_sub", {})
            if m:
                line = f"    {elem:5s}  Elem F1={m['f1']:.3f}"
                if m_sub:
                    line += f"  Sub Macro F1={m_sub['macro_f1']:.3f}"
                line += f"  (support={m['support']})"
                print(line)

        elem_macro = task11_metrics.get(valence, {}).get("elem_macro_f1", 0)
        sub_macro = task11_metrics.get(valence, {}).get("sub_macro_f1", 0)
        print(f"    ───")
        print(f"    Elem Macro F1    = {elem_macro:.3f}")
        print(f"    Sub Macro F1     = {sub_macro:.3f}  << RANKING")

    overall_elem = task11_metrics.get("overall_elem_macro_f1", 0)
    overall_sub = task11_metrics.get("overall_sub_macro_f1", 0)
    print(f"\n  Overall Elem Macro F1  = {overall_elem:.3f}")
    print(f"  *** RANKING Task 1.1   = {overall_sub:.3f} ***")

    # ── Task 1.2 ──
    print("\n── Task 1.2: Presence Rating ──────────────────────────────────")
    for valence in ("adaptive", "maladaptive"):
        m = task12_metrics.get(valence, {})
        if m and m.get("rmse") is not None:
            print(
                f"  {valence.upper():12s}"
                f"  MAE={m['mae']:.3f}"
                f"  RMSE={m['rmse']:.3f}"
                f"  QWK={m['qwk']:.3f}"
                f"  Spearman={m['spearman']:.3f}"
                f"  (n={m['n']})"
            )
    combined = task12_metrics.get("combined_rmse")
    if combined:
        print(f"\n  *** RANKING Task 1.2   = {combined:.3f} (RMSE, lower is better) ***")
    print("=" * 70)


# ── Experiment logging ────────────────────────────────────────────────────────

def log_experiment(
    task11_metrics: dict,
    task12_metrics: dict,
    config: dict,
    notes: str = "",
    log_file: str = "experiments.jsonl",
):
    """Append experiment record to JSONL log and print TSV for Google Sheet."""
    import json
    import socket
    import subprocess
    from datetime import datetime

    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # Per-element breakdown
    per_element = {}
    for valence in ("adaptive", "maladaptive"):
        for elem in ELEMENTS:
            m = task11_metrics.get(valence, {}).get(elem, {})
            m_sub = task11_metrics.get(valence, {}).get(f"{elem}_sub", {})
            per_element[f"{valence}_{elem}"] = {
                "elem_f1": round(m.get("f1", 0), 3),
                "sub_f1": round(m_sub.get("macro_f1", 0), 3),
            }

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": config.get("model", "unknown"),
        "rag_k": config.get("rag_k"),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
        "prompt_version": git_hash,
        "num_posts": config.get("num_posts"),
        "hostname": socket.gethostname(),
        "inference_time_s": config.get("inference_time_s"),
        "raw_results_file": config.get("raw_results_file", ""),
        "ranking_sub_f1": round(task11_metrics.get("overall_sub_macro_f1", 0), 3),
        "ranking_rmse": round(task12_metrics.get("combined_rmse", 0), 3),
        "elem_macro_f1": round(task11_metrics.get("overall_elem_macro_f1", 0), 3),
        "qwk_adaptive": round(task12_metrics.get("adaptive", {}).get("qwk", 0), 3),
        "qwk_maladaptive": round(task12_metrics.get("maladaptive", {}).get("qwk", 0), 3),
        "spearman_adaptive": round(task12_metrics.get("adaptive", {}).get("spearman", 0), 3),
        "spearman_maladaptive": round(task12_metrics.get("maladaptive", {}).get("spearman", 0), 3),
        "per_element": per_element,
        "notes": notes,
    }

    # Append to JSONL
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nExperiment logged to {log_file}")

    # Print CSV for Google Sheet
    csv_parts = [
        record["timestamp"][:10],
        record["model"],
        str(record["rag_k"] or ""),
        f"{record['ranking_sub_f1']:.3f}",
        f"{record['ranking_rmse']:.3f}",
        record["notes"],
    ] + [f"{v['sub_f1']:.3f}" for v in per_element.values()]
    print(f"Google Sheet (copy-paste):\n{', '.join(csv_parts)}")
