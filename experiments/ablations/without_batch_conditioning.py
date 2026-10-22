from pathlib import Path

from experiments.runner import run_ablation
from multifidelity_gp.data_io import ROOT


def run(seed, repeat=0, output=ROOT / "results" / "ablation" / "sc1095"):
    return run_ablation("without_batch_conditioning", seed, repeat, Path(output))
