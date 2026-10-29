"""Apply the frozen scenario-level comparison; never select from partial results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean


def contrast(candidate, control, *, bootstrap_samples=20000, seed=20261028):
    """Paired scenario bootstrap. Negative score deltas favour the candidate."""
    deltas = [a - b for a, b in zip(candidate, control, strict=True)]
    rng = random.Random(seed)
    distribution = sorted(fmean(rng.choices(deltas, k=len(deltas)))
                          for _ in range(bootstrap_samples))

    def quantile(q):
        at = (len(distribution) - 1) * q
        lo = int(at)
        hi = min(lo + 1, len(distribution) - 1)
        return distribution[lo] + (at - lo) * (distribution[hi] - distribution[lo])

    wins = sum(d < 0 for d in deltas)
    losses = sum(d > 0 for d in deltas)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return dict(mean_delta=fmean(deltas), ci95=[quantile(.025), quantile(.975)],
                wins=wins, losses=losses, ties=len(deltas) - n,
                sign_test_two_sided_p=p, per_scenario_delta=deltas,
                bootstrap_samples=bootstrap_samples, bootstrap_seed=seed)


def analyse(protocol, rows):
    scenarios, policies = protocol['validation_scenarios'], protocol['policies']
    if len(set(scenarios)) != len(scenarios) or len(set(policies)) != len(policies):
        raise ValueError('Protocol contains duplicate cells')
    hrc = [p for p in policies if p.startswith('hrc:')]
    ar = [p for p in policies if p.startswith('ar:')]
    if len(hrc) != 3 or len(ar) != 1 or set(policies) != {'zt_best', *hrc, *ar}:
        raise ValueError('Expected PI, exactly three reservoir seeds and one AR control')
    seeds = [next((x[2:] for x in p.split(':') if x.startswith('s=')), None) for p in hrc]
    if set(seeds) != {'0', '1', '2'}:
        raise ValueError('Expected the three frozen distinct seeds')
    expected = {(s, p) for s in scenarios for p in policies}
    observed, issues = {}, []
    for row in rows:
        key = (row.get('scenario'), row.get('policy'))
        if key not in expected:
            issues.append({'cell': key, 'error': 'unexpected cell'})
            continue
        if key in observed:
            issues.append({'cell': key, 'error': 'duplicate cell; no retrospective best-attempt selection'})
            continue
        observed[key] = row
        try:
            if row.get('error'):
                raise ValueError('recorded runtime error: ' + row['error'])
            if int(row['steps']) != protocol['steps']:
                raise ValueError('wrong or incomplete step count')
            score, reward = float(row['score']), float(row['avg_reward'])
            if not math.isfinite(score) or not math.isfinite(reward) or score < 0:
                raise ValueError('nonfinite or negative score')
            if not math.isclose(score, abs(reward), rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError('score differs from absolute mean reward')
        except (ValueError, KeyError, TypeError) as exc:
            issues.append({'cell': key, 'error': str(exc)})
    missing = sorted(expected - observed.keys())
    base = dict(expected_cells=len(expected), observed_cells=len(observed),
                missing_cells=missing, issues=issues,
                promote_hrc=False, reservoir_advantage=False)
    if missing or issues:
        return dict(base, status='incomplete_or_invalid',
                    note='No effect estimates or promotion from a partial/failed panel. Raw rows are preserved.')
    per_scenario = []
    for s in scenarios:
        seed_scores = [float(observed[s, p]['score']) for p in hrc]
        per_scenario.append(dict(scenario=s, pi=float(observed[s, 'zt_best']['score']),
                                 ar=float(observed[s, ar[0]]['score']),
                                 hrc_mean=fmean(seed_scores), hrc_seed_scores=seed_scores))
    comparisons = {}
    for name, a, b in [('hrc_vs_pi', 'hrc_mean', 'pi'), ('hrc_vs_ar', 'hrc_mean', 'ar'),
                       ('ar_vs_pi', 'ar', 'pi')]:
        comparisons[name] = contrast([r[a] for r in per_scenario], [r[b] for r in per_scenario])
    passed = lambda c: c['mean_delta'] < 0 and c['ci95'][1] < 0
    return dict(base, status='complete', per_scenario=per_scenario, comparisons=comparisons,
                # The AR attribution check uses the same interval rule as promotion.
                promote_hrc=passed(comparisons['hrc_vs_pi']),
                reservoir_advantage=passed(comparisons['hrc_vs_pi']) and passed(comparisons['hrc_vs_ar']),
                uncertainty_scope='Descriptive scenario bootstrap on this frozen panel; scenarios can share buildings/weather. '
                                  'No multiplicity adjustment or claim of generalisation to hidden competition cases.')


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1] / 'VALIDATION_PROTOCOL.json')
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    with args.csv.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = analyse(protocol, rows)
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.protocol, args.csv)}
    result['raw_rows'] = rows  # Includes errors and every seed, even when analysis is blocked.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError('Preserve previous analysis; choose a new output path')
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('status', 'expected_cells', 'observed_cells',
                                            'promote_hrc', 'reservoir_advantage')}))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
