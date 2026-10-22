from .acquisition import select_batch
from .initial_design import d_optimal_design, geometric_design
from .training import fit_models


class FusionSession:
    def __init__(self, low, candidates, initial_count, maximum_count,
                 batch_size=4, response_mode='residual', normalize=True,
                 condition_batch=True, geometric=False, random=False,
                 rng=None, restarts=2):
        self.low = low
        self.candidates = candidates
        self.maximum_count = maximum_count
        self.batch_size = batch_size
        self.response_mode = response_mode
        self.normalize = normalize
        self.condition_batch = condition_batch
        self.random = random
        self.rng = rng
        self.restarts = restarts
        design = geometric_design if geometric else d_optimal_design
        self.pending = design(low, candidates, initial_count)
        self.known = []
        self.models = {}

    def submit(self, responses):
        self.known.extend(self.pending)
        self.models, records = fit_models(self.low, responses, self.known,
                                         self.response_mode, self.restarts)
        remaining = self.maximum_count - len(self.known)
        batch = None
        if remaining:
            self.pending, batch = select_batch(
                self.low, self.models, self.known, self.candidates,
                min(self.batch_size, remaining), self.normalize,
                self.condition_batch, self.random, self.rng)
        else:
            self.pending = []
        return records, batch
