"""
Merge logic for CLPsych 2026 Task 1.

Two stages, used by all submission and ablation notebooks:
  1. build_within_model  — combine one model's P1/P2/P3 call outputs into a
                            single per-model prediction.
  2. merge_ensemble_s3 / merge_ensemble_union — combine two models' per-model
                            predictions into the final ensemble prediction.
"""


def build_within_model(unified, adaptive=None, per_a=None):
    """Combine raw call outputs into a single-model prediction.
    Mirrors the submission pipeline: maladaptive + adaptive-presence from unified;
    adaptive elements from P2 if available; Affect overridden by P3 if available."""
    merged = {
        'maladaptive': {'elements': dict(unified['maladaptive']['elements']),
                        'presence': unified['maladaptive']['presence']},
        'adaptive': {'elements': dict(adaptive['adaptive']['elements']) if adaptive else dict(unified['adaptive']['elements']),
                     'presence': unified['adaptive']['presence']},
    }
    if per_a is not None:
        for v in ('adaptive', 'maladaptive'):
            merged[v]['elements'].pop('A', None)
            if 'A' in per_a[v]['elements']:
                merged[v]['elements']['A'] = per_a[v]['elements']['A']
    for v in ('adaptive', 'maladaptive'):
        if not merged[v]['elements']:
            merged[v]['presence'] = 1
    return merged


def merge_ensemble_s3(qpred, mpred):
    """S3-merge (Submission 3): Qwen elements only, averaged presence (with the {1,2}=>2 rule)."""
    merged = {'maladaptive': {'elements': {}, 'presence': 1},
              'adaptive': {'elements': {}, 'presence': 1}}
    for v in ('maladaptive', 'adaptive'):
        pq, pg = qpred[v]['presence'], mpred[v]['presence']
        if pq == pg: merged[v]['presence'] = pq
        elif {pq, pg} == {1, 2}: merged[v]['presence'] = 2
        else: merged[v]['presence'] = round((pq + pg) / 2)
        merged[v]['elements'] = dict(qpred[v]['elements'])
        if not merged[v]['elements']: merged[v]['presence'] = 1
    return merged


def merge_ensemble_union(qpred, mpred):
    """Union-merge (Submission 2): union of both models' elements (Mistral wins
    on disagreement), averaged presence (with the {1,2}=>2 rule)."""
    merged = {'maladaptive': {'elements': {}, 'presence': 1},
              'adaptive': {'elements': {}, 'presence': 1}}
    for v in ('maladaptive', 'adaptive'):
        pq, pg = qpred[v]['presence'], mpred[v]['presence']
        if pq == pg: merged[v]['presence'] = pq
        elif {pq, pg} == {1, 2}: merged[v]['presence'] = 2
        else: merged[v]['presence'] = round((pq + pg) / 2)
        for e in set(qpred[v]['elements'].keys()) | set(mpred[v]['elements'].keys()):
            sq, sm = qpred[v]['elements'].get(e), mpred[v]['elements'].get(e)
            if sq == sm:
                merged[v]['elements'][e] = sq
            else:
                # disagreement: Mistral wins if it made a prediction, else fall back to Qwen's
                merged[v]['elements'][e] = sm if sm is not None else sq
        if not merged[v]['elements']: merged[v]['presence'] = 1
    return merged
