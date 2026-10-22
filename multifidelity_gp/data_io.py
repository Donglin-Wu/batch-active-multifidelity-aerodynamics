import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('cl', 'cd', 'cm')
COORDINATES = ['alpha', 'mach']


def read_frame(path):
    return pd.read_csv(path, float_precision='round_trip').sort_values(
        COORDINATES).reset_index(drop=True)


def load_data(case):
    directory = ROOT / 'data' / case
    return (read_frame(directory / 'low_fidelity.csv'),
            read_frame(directory / 'high_fidelity.csv'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':')), encoding='utf-8')
