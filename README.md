# HRC-PI: online reservoir forecasts for transferable HVAC control

An entry under development for the [NeurIPS 2026 Smart Buildings Challenge](https://www.kaggle.com/competitions/neurips-2026-smart-buildings).
It combines the author's [Holographic Reservoir](https://github.com/Agnuxo1/Holographic-Reservoir)
with the credited public per-zone PI controller. The official simulator supplies
all temperatures, rewards and energy measurements. No synthetic reward model is used.

## What is implemented

The shared reservoir predicts each zone's temperature one hour ahead using only
observations available at that time. A gated feedforward term adjusts the PI error:
`error = temperature - target + kappa * gate * predicted_drift`.
The gate opens when the predictor beats persistence online; `kappa=0` exactly
recovers the PI control. Readout learning happens inside each new building;
validation scenarios never tune the offline settings.

## Measured results, 28 October

**Three experimental stages use the real official simulator.**

| Stage | Measured outcome | Scope |
|---|---|---|
| One-hour prediction study | Selected reservoir RMSE 0.255631 K vs persistence 0.342138 K | Five tune traces; a 109-row predictor grid; selection data |
| Closed-loop feedforward sweep | HRC kappa0.25 score **0.0061340**, vs PI **0.0062008** (1.08% lower) | Five tune scenarios,4032steps each; not held-out or official score |
| Linear-readout ablation | AR kappa0.25 score **0.0061337**, slightly better than HRC on tune | Same scenarios and budget; reservoir benefit not established |

The frozen prospective validation is now complete: **80/80 episodes** across
16 scenarios (10 official and six generated building/weather combinations),
8640 steps each, with three reservoir seeds averaged inside each scenario.

| Held-out panel, lower score is better | Mean score |
|---|---:|
| PI control | 0.0174040884 |
| HRC-PI, three-seed mean | **0.0173073431** |
| Linear AR control | 0.0172917679 |

HRC improves on PI by **0.56%**: paired mean difference −0.0000967453,
scenario-bootstrap 95% interval **[−0.0002038972, −0.0000083849]**;
11 scenarios improve and 5 regress. This passes the pre-registered PI
comparison gate. The two-sided sign test is 0.2101, so the evidence depends
on effect sizes rather than a significant win-count test.

HRC versus AR has mean difference +0.0000155752 and interval
[−0.0000376547, +0.0000882530]. **Reservoir-specific superiority over AR is
not established.** AR versus PI also has an interval crossing zero. These
comparisons do not invalidate the measured HRC-versus-PI improvement or
the earlier forecasting result. [All raw episodes and fixed analysis](results/validation-20261029).

The predictor improvement and small tune control gain are positive measured
results; the AR comparison and the finite scenario panel limit the conclusions.
They do not invalidate the forecasting gain or establish a leaderboard gain.

## Frozen prospective validation

[`VALIDATION_PROTOCOL.json`](VALIDATION_PROTOCOL.json) fixes kappa0.25,
delta0.1,lambda0.99999,N150 and reservoir seeds0,1,2 before validation.
Compare all three seeds, the linear AR ablation and PI on the same16scenarios.
Average seeds within each scenario, then report paired score differences,
scenario-bootstrap intervals and sign tests. HRC is promoted only if it beats
PI with an upper95%interval below zero. Reservoir attribution also needs a
paired advantage over AR. Failed scenarios remain in the record.

The [analysis command](experiments/analyse_validation.py) is fixed while the
managed 80-episode run was in progress. It rejects missing, duplicate, failed,
nonfinite or short episodes; it never promotes from a partial panel. Every raw
row and each seed's result are retained. Run it after downloading the job:

```sh
python experiments/analyse_validation.py --csv output/validation.csv --out output/analysis.json
```

The bootstrap unit is a scenario after averaging reservoir seeds. Scenarios
can share buildings or weather, so the interval describes this fixed panel;
it does not establish performance on hidden competition cases. Intervals are
not adjusted for multiple comparisons. Reservoir attribution uses the same
negative-upper-interval rule against AR as promotion uses against PI.

The score is the absolute mean per-step reward; **lower is better**. A one-hour
forecast gain is not a controller or competition score. No final submission,
official score or paper acceptance is claimed for this repository.

## Reproduce

Install the [official starter kit](https://github.com/smart-buildings-2026/starter-kit)
and this project's `requirements.txt`. Use `experiments/run_eval.py --help` for
explicit kit,scenario,policy,step and worker arguments. `run_validation.py` is
the managed server job: it copies the official package into an isolated working
directory, generates the declared unseen scenarios and runs the fixed protocol.
It does not modify the shared kit. CPU work uses the server's managed queue,
four processes with one numerical-library thread each. No local GPU is used.

The official Kaggle API currently gives final deadline **15 January2027,
17:59:59UTC (18:59:59Madrid)**; earlier paper/entry deadlines must be checked
separately. Results here are simulator measurements and work in progress.

## Attribution and licence

Apache-2.0. See [NOTICE](NOTICE) for the author's Holographic-Reservoir,
Malyshev Danil's public ZT Best controller and the official starter kit.
Competition data and the starter kit are not redistributed. The scenario
definitions are our generated validation inputs, not hidden competition cases.
