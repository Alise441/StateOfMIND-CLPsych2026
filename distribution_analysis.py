"""
Compare distributions: train gold labels vs test predictions.

Usage:
  python distribution_analysis.py outputs/raw_results_test.json
  python distribution_analysis.py outputs/raw_results_test.json -o outputs/distribution_analysis.txt
"""

import argparse
import io
import json
import re
import sys
from collections import Counter

from data_loader import load_all_timelines
from taxonomy import ELEMENTS, SUBELEMENTS


def gold_from_post(post):
    def parse_state(state):
        elements = {}
        for attr, key in [('A','A'),('B_O','B-O'),('B_S','B-S'),('C_O','C-O'),('C_S','C-S'),('D','D')]:
            ev = getattr(state, attr)
            if ev is not None:
                m = re.match(r'\((\d+)\)', ev.category.strip())
                if m:
                    elements[key] = int(m.group(1))
        return {'presence': state.presence, 'elements': elements}
    return {'adaptive': parse_state(post.adaptive_state), 'maladaptive': parse_state(post.maladaptive_state)}


def get_presence_counts(data, source):
    counts = {'adaptive': Counter(), 'maladaptive': Counter()}
    for item in data:
        d = gold_from_post(item) if source == 'gold' else item['prediction']
        for v in ('adaptive', 'maladaptive'):
            p = d[v].get('presence')
            if p is not None:
                counts[v][p] += 1
    return counts


def get_elem_sub_counts(data, source):
    elem_counts = Counter()
    sub_counts = Counter()
    for item in data:
        d = gold_from_post(item) if source == 'gold' else item['prediction']
        for valence in ('adaptive', 'maladaptive'):
            for elem, sub_id in d[valence].get('elements', {}).items():
                elem_counts[(valence, elem)] += 1
                sub_name = SUBELEMENTS.get(elem, {}).get(valence, {}).get(sub_id, '?')
                sub_short = sub_name.split(' — ')[0][:25] if isinstance(sub_name, str) else str(sub_name)
                sub_counts[(valence, elem, sub_id, sub_short)] += 1
    return elem_counts, sub_counts


def run(pred_file, output=None):
    # Suppress load messages
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    train_posts = load_all_timelines('train_tasks12')
    sys.stdout = old_stdout

    train_evidenced = [p for p in train_posts if p.has_evidence]

    with open(pred_file) as f:
        test_results = json.load(f)
    test_with_preds = [r for r in test_results
        if r['prediction']['adaptive']['elements'] or r['prediction']['maladaptive']['elements']]

    train_n = len(train_evidenced)
    test_all_n = len(test_results)
    test_filt_n = len(test_with_preds)

    train_pres = get_presence_counts(train_evidenced, 'gold')
    test_all_pres = get_presence_counts(test_results, 'pred')
    test_filt_pres = get_presence_counts(test_with_preds, 'pred')
    train_elem, train_sub = get_elem_sub_counts(train_evidenced, 'gold')
    test_all_elem, test_all_sub = get_elem_sub_counts(test_results, 'pred')
    test_filt_elem, test_filt_sub = get_elem_sub_counts(test_with_preds, 'pred')

    out = open(output, 'w') if output else sys.stdout

    print(f"# Distribution Analysis: Train Gold vs Test Predictions", file=out)
    print(f"# Source: {pred_file}", file=out)
    print(f"# Train: {train_n} evidenced posts (gold labels)", file=out)
    print(f"# Test (all): {test_all_n} posts, Test (with predictions): {test_filt_n} posts", file=out)
    print(file=out)

    print("## Element Presence Rate", file=out)
    print(file=out)
    print(f"{'Element':30s} {'Train gold':>10s} {'Test all':>10s} {'Test filt':>10s} {'D(filt)':>8s}", file=out)
    print("-" * 72, file=out)
    for valence in ('adaptive', 'maladaptive'):
        for elem in ELEMENTS:
            tr = train_elem[(valence, elem)] / train_n * 100
            ta = test_all_elem[(valence, elem)] / test_all_n * 100
            tf = test_filt_elem[(valence, elem)] / test_filt_n * 100
            print(f"{valence + ' ' + elem:30s} {tr:9.1f}% {ta:9.1f}% {tf:9.1f}% {tf-tr:+7.1f}%", file=out)

    print(file=out)
    print("## Presence Distribution", file=out)
    for valence in ('adaptive', 'maladaptive'):
        print(file=out)
        print(f"### {valence.upper()}", file=out)
        print(f"{'Score':6s} {'Train gold':>10s} {'Test all':>10s} {'Test filt':>10s}", file=out)
        for score in range(1, 6):
            tr = train_pres[valence][score] / train_n * 100
            ta = test_all_pres[valence][score] / test_all_n * 100
            tf = test_filt_pres[valence][score] / test_filt_n * 100
            print(f"  {score}    {tr:9.1f}% {ta:9.1f}% {tf:9.1f}%", file=out)

    print(file=out)
    print("## Subelement Distribution", file=out)
    print(file=out)
    all_keys = set(list(train_sub.keys()) + list(test_filt_sub.keys()))
    rows = []
    for key in all_keys:
        tr = train_sub.get(key, 0)
        tf = test_filt_sub.get(key, 0)
        label = f"{key[0]} {key[1]}:{key[2]} ({key[3]})"
        rows.append((label, tr, tf, tr/train_n*100, tf/test_filt_n*100))
    rows.sort(key=lambda x: -x[1])

    print(f"{'Subelement':55s} {'Train':>5s} {'Tr%':>6s} {'Test':>5s} {'Te%':>6s} {'D':>7s}", file=out)
    print("-" * 88, file=out)
    for label, tr, tf, tr_pct, tf_pct in rows:
        print(f"{label:55s} {tr:5d} {tr_pct:5.1f}% {tf:5d} {tf_pct:5.1f}% {tf_pct-tr_pct:+6.1f}%", file=out)

    if output:
        out.close()
        print(f"Saved to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare train gold vs test prediction distributions")
    parser.add_argument('pred_file', help='Path to raw_results JSON file')
    parser.add_argument('-o', '--output', help='Output file (default: print to stdout)')
    args = parser.parse_args()
    run(args.pred_file, args.output)
