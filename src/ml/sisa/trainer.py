import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.ml.common.metrice import Accuracy
from src.ml.common.model import TrainSummary
from src.ml.sisa.model import SessionInterestSelfAttention
from src.utils import progressbar


class SisaTrainer:

    def __init__(self, model: SessionInterestSelfAttention, learning_rate: float):
        super().__init__()
        self.model = model

        self.optim = AdamW(params=model.parameters(), lr=learning_rate)
        self.cross_entropy = nn.BCELoss()
        self.metrics = [Accuracy()]

    def _backpropagation(self, loss: Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()

    def _inference(self, output, session, padding_mask, target, src_mask, label):
        pred = self.model(
            session_items=session, padding_mask=padding_mask, target_item=target, src_mask=src_mask
        )
        loss = self.cross_entropy(pred, label)
        pred = torch.where(pred > 0.5, 1, 0)
        output['y_pred'].extend(pred.squeeze(0).cpu().tolist())
        output['y_true'].extend(label.squeeze(0).cpu().tolist())
        return loss

    def train(self, train_loader: DataLoader, summary: TrainSummary) -> TrainSummary:
        src_mask = generate_square_subsequent_mask(sz=20)
        total = len(train_loader)

        output = {'y_pred': [], 'y_true': []}
        for step, (session, padding_mask, target, label) in enumerate(train_loader, start=1):
            progressbar(total=total, i=step, prefix='train')
            loss = self._inference(output, session, padding_mask, target, src_mask, label)
            self._backpropagation(loss=loss)
            summary.add_train_loss(loss.item())

        for func in self.metrics:
            value = func(output['y_pred'], output['y_true'])
            summary.add_train_metrics(func=func, value=value)
        return summary

    def val(self, val_loader: DataLoader, summary: TrainSummary) -> TrainSummary:
        src_mask = generate_square_subsequent_mask(sz=20)

        output = {'y_pred': [], 'y_true': []}
        with torch.no_grad():
            for step, (session, padding_mask, target, label) in enumerate(val_loader, start=1):
                loss = self._inference(output, session, padding_mask, target, src_mask, label)
                summary.add_val_loss(loss.item())

        for func in self.metrics:
            value = func(output['y_pred'], output['y_true'])
            summary.add_val_metrics(func=func, value=value)
        return summary

    def test(self, test_loader: DataLoader) -> None:
        src_mask = generate_square_subsequent_mask(sz=20)

        output = {'y_pred': [], 'y_true': []}
        test_loss = 0
        with torch.no_grad():
            for step, (session, padding_mask, target, label) in enumerate(test_loader, start=1):
                loss = self._inference(output, session, padding_mask, target, src_mask, label)
                test_loss += loss.item()

        print(f'test result')
        print(f'\ttest_loss: {test_loss / step:0.3f}')
        for func in self.metrics:
            value = func(output['y_pred'], output['y_true'])
            print(f'\ttest_{func}: {value: 0.3f}')

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
