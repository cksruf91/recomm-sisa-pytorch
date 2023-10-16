import sys
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data_iterator import DataIterator
from src.ml.model.sisa import SessionInterestSelfAttention


class Metrics(metaclass=ABCMeta):

    @abstractmethod
    def __str__(self):
        """ return display name """
        pass


class Accuracy(Metrics):

    def __str__(self):
        return "acc"

    def __call__(self, output: list, label: list) -> float:
        """get accuracy
        Args:
            output : model prediction, dim: [total data size]
            label : label, dim: [total data size]
        Returns:
            float: accuracy
        """
        assert len(output) == len(label)

        total = len(output)
        label_array = np.array(label)
        output_array = np.array(output)

        match = np.sum(label_array == output_array)
        return match / total


@dataclass
class TrainSummary:
    epoch: int = field()
    start_time: int = field(default_factory=time.time)

    train_loss: float = field(default=0.)
    train_step: int = field(default=0)

    val_loss: float = field(default=0.)
    val_step: int = field(default=0)

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

    def add_train_metrics(self, func: Metrics, value: Any):
        self.metrics[f'train_{func}'] = value

    def add_val_metrics(self, func: Metrics, value: Any):
        self.metrics[f'val_{func}'] = value

    def __repr__(self):
        summary = (f'epoch {self.epoch:02d} loss: {self.train_loss / self.train_step:3.3f} '
                   f'val_loss: {self.val_loss / self.val_step:3.3f}')
        for k, v in self.metrics.items():
            summary += f" {k} : {v:3.3f}"
        return summary


class Train:

    def __init__(self, model: SessionInterestSelfAttention, epoch: int = 10):
        self.model = model
        self.epoch = epoch

        self.optim = AdamW(params=model.parameters(), lr=0.005)
        self.cross_entropy = nn.BCELoss()
        self.metrics = [Accuracy()]

    def _backpropagation(self, loss: Tensor):
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()

    def run(self, train_data: DataIterator, val_data: DataIterator):
        print('start training')
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        for e in range(self.epoch):
            summary = TrainSummary(epoch=e + 1)
            summary = self._train(train_loader=train_loader, summary=summary)
            summary = self._val(val_loader=val_loader, summary=summary)
            print(summary)

    def _train(self, train_loader: DataLoader, summary: TrainSummary):
        src_mask = generate_square_subsequent_mask(sz=20)
        total = len(train_loader)

        output = {'y_pred': [], 'y_true': []}
        for step, (session, padding_mask, target, label) in enumerate(train_loader, start=1):
            progressbar(total=total, i=step, prefix='train')
            pred = self.model(
                session_items=session, padding_mask=padding_mask, target_item=target, src_mask=src_mask
            )
            loss = self.cross_entropy(pred, label)
            self._backpropagation(loss=loss)

            summary.add_train_loss(loss.item())
            pred = torch.where(pred > 0.5, 1, 0)
            output['y_pred'].extend(pred.squeeze(0).cpu().tolist())
            output['y_true'].extend(label.squeeze(0).cpu().tolist())

        for func in self.metrics:
            value = func(output['y_pred'], output['y_true'])
            summary.add_train_metrics(func=func, value=value)
        return summary

    def _val(self, val_loader: DataLoader, summary: TrainSummary):
        src_mask = generate_square_subsequent_mask(sz=20)

        output = {'y_pred': [], 'y_true': []}
        with torch.no_grad():
            for step, (session, padding_mask, target, label) in enumerate(val_loader, start=1):
                pred = self.model(
                    session_items=session, padding_mask=padding_mask, target_item=target, src_mask=src_mask
                )
                loss = self.cross_entropy(pred, label)
                summary.add_val_loss(loss.item())
                pred = torch.where(pred > 0.5, 1., 0.)
                output['y_pred'].extend(pred.squeeze(0).cpu().tolist())
                output['y_true'].extend(label.squeeze(0).cpu().tolist())

        for func in self.metrics:
            value = func(output['y_pred'], output['y_true'])
            summary.add_val_metrics(func=func, value=value)
        return summary

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def progressbar(total, i, bar_length=50, prefix='', suffix=''):
    """progressbar
    """
    bar_graph = '█'
    if i % max((total // 100), 1) == 0:
        dot_num = int((i + 1) / total * bar_length)
        dot = bar_graph * dot_num
        empty = '.' * (bar_length - dot_num)
        sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')
    if i == total:
        sys.stdout.write(f'\r {prefix} [{bar_graph * bar_length}] {100:3.2f}% {suffix}')
        print(' Done')
