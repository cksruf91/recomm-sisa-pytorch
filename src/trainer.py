from torch.utils.data import DataLoader

from src.ml.common.iterator import DataIterator
from src.ml.common.model import TrainSummary
from src.ml.common.callback import EarlyStopping
from src.ml.deep_fm.trainer import DeepFmTrainer
from src.ml.sisa.trainer import SisaTrainer


class ModelTrainer:

    def __init__(self, *args, **kwargs):
        self.arch: str = kwargs.pop('arch')
        self.epoch: int = kwargs.pop('epoch', 50)
        if self.arch == 'sisa':
            self.engine = SisaTrainer(*args, **kwargs)
        elif self.arch == 'deepfm':
            self.engine = DeepFmTrainer(*args, **kwargs)
        else:
            raise ValueError(f'arch [{self.arch}] not Implemented')

    def run(self, train_data: DataIterator, val_data: DataIterator, early_stop: int = 10, monitor: str = 'val_acc'):
        print('start training')
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        early_stopping = EarlyStopping(patience=early_stop, monitor=monitor, mode='max')
        for e in range(self.epoch):
            summary = TrainSummary(epoch=e + 1)
            summary = self.engine.train(train_loader=train_loader, summary=summary)
            summary = self.engine.val(val_loader=val_loader, summary=summary)
            print(summary)

            if early_stopping(summary):
                print('early stop training')
                break

        return self

    def test(self, test_data: DataIterator):
        print('test model')
        test_data = DataLoader(test_data, batch_size=32, shuffle=True)
        summary = self.engine.train(train_loader=test_data, summary=summary)
