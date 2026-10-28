"""Managed CPU job: isolated official kit, fixed validation, no model tuning."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
FROZEN = json.loads((ROOT / 'VALIDATION_PROTOCOL.json').read_text())
OFFICIAL = Path('/home/ai/proyectos/smart-buildings/starter-kit')
KIT = ROOT / 'isolated-kit'
if not KIT.exists():
    shutil.copytree(OFFICIAL / 'smart_buildings', KIT / 'smart_buildings',
                    ignore=shutil.ignore_patterns('__pycache__', '.git', '.venv'))
for path in (ROOT / 'scenarios/unseen_buildings').glob('*.yaml'):
    shutil.copy2(path, KIT / 'smart_buildings/buildings' / path.name)
unseen = KIT / 'smart_buildings/unseen_scenarios'
for name in (ROOT / 'experiments/splits/validation_unseen.txt').read_text().splitlines():
    building, weather = name.split('/')
    if not (unseen / name / 'config.gin').exists():
        subprocess.run([sys.executable, '-m', 'smart_buildings.run_scenario_generator',
                        '--buildings', building, '--weathers', weather,
                        '--output-dir', str(unseen)], cwd=KIT, check=True)
out = ROOT / 'output'; out.mkdir(exist_ok=True)
shutil.copy2(ROOT / 'VALIDATION_PROTOCOL.json', out / 'frozen_protocol.json')
subprocess.run([sys.executable, str(ROOT / 'experiments/run_eval.py'), '--kit', str(KIT),
                '--roots', str(KIT / 'smart_buildings/training_scenarios'), str(unseen),
                '--scenarios', str(ROOT / 'experiments/splits/validation.txt'),
                '--policies', *FROZEN['policies'], '--steps', str(FROZEN['steps']),
                '--workers', str(FROZEN['workers']), '--out', str(out / 'validation.csv')], check=True)
