"""Summarize an evaluate.sh run: unbiased pass@k over reps, plus per-repo and failure-mode breakdowns.

usage: python3 eval/passk.py runs/NAME [runs/NAME2 ...]   (worker dirs r<rep>_s<shard>; several runs = more reps)
"""
import collections
import glob
import json
import math
import os
import re
import statistics
import sys

res = collections.defaultdict(dict)  # task -> (run, rep) -> resolved
rows = []
for run in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(run, 'r*_s*'))):
        m = re.fullmatch(r'r(\d+)_s\d+', os.path.basename(d))
        f = os.path.join(d, 'results.jsonl')
        if not m or not os.path.exists(f):
            continue
        for x in map(json.loads, open(f)):
            res[x['id']][(run, m.group(1))] = bool(x['resolved'])
            rows.append(x)
if not rows:
    sys.exit('no results.jsonl found')


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


ns = [len(v) for v in res.values()]
print(f'tasks {len(res)}  attempts {len(rows)}  attempts/task {min(ns)}-{max(ns)}  '
      f'resolved {sum(r["resolved"] for r in rows)}')
for k in (1, 2, 4, 8, 16):
    vals = [pass_at_k(len(v), sum(v.values()), k) for v in res.values() if len(v) >= k]
    if vals:
        print(f'pass@{k:<2} = {sum(vals) / len(vals):.3f}  ({len(vals)} tasks with >= {k} attempts)')
by_repo = collections.defaultdict(list)
for t, v in res.items():
    by_repo[t.rsplit('_', 1)[0]].append(sum(v.values()) / len(v))
print('per repo pass@1: ' + ', '.join(f'{p} {sum(x) / len(x):.3f} ({len(x)})' for p, x in sorted(by_repo.items())))
med = lambda k: statistics.median(r[k] for r in rows)
print(f'median tool calls {med("tool_calls")}, median duration {med("duration_s"):.0f}s, '
      f'no patch {sum(not r["patch_chars"] for r in rows) / len(rows):.0%}')
