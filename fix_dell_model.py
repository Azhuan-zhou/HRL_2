"""Make a Dell-trained ACT model loadable by stock LeRobot on the Mac.
Dell's LeRobot fork adds tactile options (off for these models); stock LeRobot rejects the keys.
Strips every key containing 'tactile' from the json files rollout reads; originals kept as *.dell.json.
Usage: python fix_dell_model.py ~/HRL_models/v3_c30x10 [more dirs...]
"""
import json, shutil, sys
from pathlib import Path


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if 'tactile' not in k.lower()}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


for d in map(Path, sys.argv[1:]):
    for name in ['config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
        f, bak = d / name, d / name.replace('.json', '.dell.json')
        if not bak.exists():
            shutil.copy2(f, bak)
        data = json.loads(bak.read_text())
        assert not json.loads(bak.read_text()).get('use_tactile'), f'{f}: tactile is ON, stripping would break the model'
        f.write_text(json.dumps(strip(data), indent=4))
        print('fixed', f)
