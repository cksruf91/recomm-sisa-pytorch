import numpy as np

from src.ml.common.model import TrainSummary


class EarlyStopping:

    def __init__(self, patience: int = 10, monitor: str = "val_loss", mode: str = 'min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == 'min' else -np.inf
        self._count = 0

    def _is_best(self, new: float, best: float) -> bool:
        return new <= best if self.mode == 'min' else new >= best

    def __call__(self, summary: TrainSummary) -> bool:
        score = summary[self.monitor]
        if self._is_best(score, self.best):
            print(f'\tbest score [{self.monitor}] : {self.best:0.3f} -> {score:0.3f}')
            self.best = score
            self._count = 0
        else:
            self._count += 1

        return True if self._count > self.patience else False
