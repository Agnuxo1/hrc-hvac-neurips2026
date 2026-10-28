"""Fixed, seeded split of evaluation scenarios (written before any HRC-PI run).

official 30 (training_set.yaml, 6 buildings x 5 stations):
  validation_official: 2 scenarios per station, from different buildings  (10)
  tune:                1 scenario per station, disjoint from validation     (5)
unseen buildings (generated with the official generator, not in the shipped six):
  validation_unseen: 6 of the 12 building_91xx scenarios, 2 per station, every building used
"""
import random
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent / "splits"
SEED = 20261028


def main(training_set_yaml: str) -> None:
    pairs = [(s["building"], s["weather"]) for s in yaml.safe_load(open(training_set_yaml))["scenarios"]]
    rng = random.Random(SEED)
    stations = sorted({w.rsplit("_", 1)[0] for _, w in pairs})
    val, tune = [], []
    for st in stations:
        cand = [p for p in pairs if p[1].startswith(st + "_")]
        rng.shuffle(cand)
        val += cand[:2]
        tune += cand[2:3]
    unseen = ["building_9102/fairbanks_3", "building_9102/singapore_2", "building_9106/phoenix_3",
              "building_9107/fairbanks_3", "building_9107/singapore_2", "building_9108/phoenix_3"]
    OUT.mkdir(exist_ok=True)
    (OUT / "validation_official.txt").write_text("\n".join(f"{b}/{w}" for b, w in val) + "\n")
    (OUT / "tune.txt").write_text("\n".join(f"{b}/{w}" for b, w in tune) + "\n")
    (OUT / "validation_unseen.txt").write_text("\n".join(unseen) + "\n")
    (OUT / "official30.txt").write_text("\n".join(f"{b}/{w}" for b, w in pairs) + "\n")
    (OUT / "validation.txt").write_text("\n".join(unseen + [f"{b}/{w}" for b, w in val]) + "\n")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
