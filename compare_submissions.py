"""
Compare submissions: TP/FP/FN for S1 and S2, prediction counts for S3.
"""

ELEMENTS = [
    'adaptive A', 'adaptive B-O', 'adaptive B-S',
    'adaptive C-O', 'adaptive C-S', 'adaptive D',
    'maladaptive A', 'maladaptive B-O', 'maladaptive B-S',
    'maladaptive C-O', 'maladaptive C-S', 'maladaptive D',
]

S1_DIR = 'submissions/submission1'
S2_DIR = 'submissions/submission2'
S3_DIR = 'submissions/submission3'


def parse_scoring_result(path):
    results = {}
    with open(path) as f:
        lines = f.readlines()

    in_section = False
    for line in lines:
        if 'Element Presence (Binary)' in line:
            in_section = True
            continue
        if 'Subelement Classification' in line:
            in_section = False
            continue
        if not in_section:
            continue

        line = line.strip()
        for elem_key in ['adaptive-state:A', 'adaptive-state:B-O', 'adaptive-state:B-S',
                         'adaptive-state:C-O', 'adaptive-state:C-S', 'adaptive-state:D',
                         'maladaptive-state:A', 'maladaptive-state:B-O', 'maladaptive-state:B-S',
                         'maladaptive-state:C-O', 'maladaptive-state:C-S', 'maladaptive-state:D']:
            if line.startswith(elem_key):
                parts = line.split()
                prec = float(parts[-4])
                rec = float(parts[-3])
                supp = int(parts[-1])
                v, e = elem_key.replace('-state:', ' ').split(' ', 1)
                results[f'{v} {e}'] = {'prec': prec, 'rec': rec, 'supp': supp}
    return results


def compute_tp_fp_fn(results):
    table = {}
    for elem, r in results.items():
        tp = round(r['rec'] * r['supp'])
        pred = round(tp / r['prec']) if r['prec'] > 0 else 0
        fp = pred - tp
        fn = r['supp'] - tp
        table[elem] = {'tp': tp, 'fp': fp, 'fn': fn, 'pred': pred}
    return table


def count_predictions(pred_path):
    import json
    from collections import Counter
    with open(pred_path) as f:
        data = json.load(f)
    counts = Counter()
    for entry in data:
        for vkey in ('adaptive-state', 'maladaptive-state'):
            v = 'adaptive' if vkey == 'adaptive-state' else 'maladaptive'
            st = entry.get(vkey, {})
            for k in st:
                if k != 'Presence':
                    counts[f'{v} {k}'] += 1
    return counts


def main():
    import os

    # S1 and S2: full TP/FP/FN from scoring results
    s1 = compute_tp_fp_fn(parse_scoring_result(f'{S1_DIR}/scoring_result.txt'))
    s2 = compute_tp_fp_fn(parse_scoring_result(f'{S2_DIR}/scoring_result.txt'))

    # S3: prediction counts only (no scoring)
    has_s3 = os.path.exists(f'{S3_DIR}/task1_pred_ens.json')
    s3_counts = count_predictions(f'{S3_DIR}/task1_pred_ens.json') if has_s3 else {}

    # Get gold support from S1
    s1_scoring = parse_scoring_result(f'{S1_DIR}/scoring_result.txt')
    gold = {elem: s1_scoring[elem]['supp'] for elem in ELEMENTS}

    # Header
    header = f'{"Element":20s} {"Gold":>5s} | {"S1 TP":>5s} {"FP":>4s} {"FN":>4s} {"Pred":>5s} | {"S2 TP":>5s} {"FP":>4s} {"FN":>4s} {"Pred":>5s} | {"ΔTP":>4s} {"ΔFP":>4s} {"ΔFN":>4s}'
    if has_s3:
        header += f' | {"S3 Pred (Ens)":>7s}'
    print(header)
    print('-' * len(header))

    tot1 = {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0}
    tot2 = {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0}
    tot3 = 0
    tot_gold = 0

    for elem in ELEMENTS:
        t1 = s1.get(elem, {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0})
        t2 = s2.get(elem, {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0})
        g = gold.get(elem, 0)

        dtp = t2['tp'] - t1['tp']
        dfp = t2['fp'] - t1['fp']
        dfn = t2['fn'] - t1['fn']

        line = (f'{elem:20s} {g:5d} | {t1["tp"]:5d} {t1["fp"]:4d} {t1["fn"]:4d} {t1["pred"]:5d} |'
                f' {t2["tp"]:5d} {t2["fp"]:4d} {t2["fn"]:4d} {t2["pred"]:5d} |'
                f' {dtp:+4d} {dfp:+4d} {dfn:+4d}')
        tot_gold += g

        if has_s3:
            s3p = s3_counts.get(elem, 0)
            line += f' | {s3p:7d}'
            tot3 += s3p

        print(line)

        for k in ('tp', 'fp', 'fn', 'pred'):
            tot1[k] += t1[k]
            tot2[k] += t2[k]

    print('-' * len(header))

    dtp = tot2['tp'] - tot1['tp']
    dfp = tot2['fp'] - tot1['fp']
    dfn = tot2['fn'] - tot1['fn']

    line = (f'{"TOTAL":20s} {tot_gold:5d} | {tot1["tp"]:5d} {tot1["fp"]:4d} {tot1["fn"]:4d} {tot1["pred"]:5d} |'
            f' {tot2["tp"]:5d} {tot2["fp"]:4d} {tot2["fn"]:4d} {tot2["pred"]:5d} |'
            f' {dtp:+4d} {dfp:+4d} {dfn:+4d}')

    if has_s3:
        line += f' | {tot3:7d}'

    print(line)


if __name__ == '__main__':
    main()
