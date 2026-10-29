"""Synthetic contract tests for analysis, not simulator experiments."""
from copy import deepcopy

from experiments.analyse_validation import analyse, contrast


def panel():
    policies = ['zt_best', 'hrc:s=0', 'hrc:s=1', 'hrc:s=2', 'ar:s=0']
    protocol = dict(policies=policies, validation_scenarios=['a', 'b'], steps=8640)
    rows = [dict(scenario=s, policy=p, steps=8640, score=v, avg_reward=-v, error='')
            for s in ['a', 'b'] for p, v in zip(policies, [10, 8, 9, 10, 8.5])]
    return protocol, rows


def test_seeds_are_averaged_and_ar_can_prevent_attribution():
    protocol, rows = panel()
    result = analyse(protocol, rows)
    assert result['status'] == 'complete' and result['promote_hrc']
    assert len(result['comparisons']['hrc_vs_pi']['per_scenario_delta']) == 2
    assert result['comparisons']['hrc_vs_pi']['mean_delta'] == -1
    assert result['comparisons']['hrc_vs_ar']['mean_delta'] == .5
    assert not result['reservoir_advantage']


def test_missing_failed_duplicate_and_nonfinite_cells_block_promotion():
    protocol, rows = panel()
    variants = [rows[:-1], rows + [rows[0]]]
    for field, value in [('error', 'runtime failure'), ('steps', 4000), ('score', 'nan')]:
        bad = deepcopy(rows)
        bad[0][field] = value
        variants.append(bad)
    for bad in variants:
        result = analyse(protocol, bad)
        assert result['status'] == 'incomplete_or_invalid'
        assert not result['promote_hrc']
        assert 'comparisons' not in result


def test_exact_sign_test_and_bootstrap_retain_both_directions():
    r = contrast([0, 2, 0, 2], [1, 1, 1, 1], bootstrap_samples=1000)
    assert r['wins'] == r['losses'] == 2 and r['sign_test_two_sided_p'] == 1
    assert r['ci95'][0] < 0 < r['ci95'][1]
