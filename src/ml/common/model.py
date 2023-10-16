import time
from dataclasses import dataclass, field
from typing import Dict, Any

from src.ml.common.metrice import Metrics


@dataclass
class TrainSummary:
    epoch: int = field()
    start_time: int = field(default_factory=time.time)

    train_loss: float = field(default=0.)
    train_step: int = field(default=0)
    val_loss: float = field(default=0.)
    val_step: int = field(default=0)
    test_loss: float = field(default=0.)
    test_step: int = field(default=0)

    metrics: Dict = field(default_factory=dict)

    early_stop: bool = field(default=False)

    def __getitem__(self, name: str):
        try:
            return self.metrics[name]
        except KeyError:
            return getattr(self, name)

    def add_train_loss(self, loss: float):
        self.train_loss += loss
        self.train_step += 1

    def add_val_loss(self, loss: float):
        self.val_loss += loss
        self.val_step += 1

    def add_test_loss(self, loss: float):
        self.test_loss += loss
        self.test_step += 1

    def add_train_metrics(self, func: Metrics, value: Any):
        self.metrics[f'train_{func}'] = value

    def add_val_metrics(self, func: Metrics, value: Any):
        self.metrics[f'val_{func}'] = value

    def add_test_metrics(self, func: Metrics, value: Any):
        self.metrics[f'val_{func}'] = value

    def __repr__(self):
        summary = (f'epoch {self.epoch:02d} loss: {self.train_loss / self.train_step:3.3f} '
                   f'val_loss: {self.val_loss / self.val_step:3.3f}')
        for k, v in self.metrics.items():
            summary += f" {k} : {v:3.3f}"
        return summary
