import pandas as pd
import torch
from torch.utils.data import Dataset

from setting import Config

CONFIG = Config()


class DataIterator(Dataset):

    def __init__(self, dataset: pd.DataFrame):
        self._parsing_dataset(dataset)

    def __len__(self):
        return self._length

    def _parsing_dataset(self, dataset: pd.DataFrame):
        dataset['UserID'] = dataset['UserID'].astype(int)
        dataset['SessionID'] = dataset['SessionID'].astype(int)
        dataset['Target'] = dataset['Target'].astype(int)
        dataset['Session'] = dataset['Session'].map(eval)
        dataset['Negative'] = dataset['Negative'].map(eval)

        self.dataset = []
        for row in dataset.to_dict(orient='records'):
            self.dataset.append(self._parsing_row(row))
        self._length: int = len(self.dataset)

    def _parsing_row(self, row):
        raise NotImplementedError('_parsing_row is not implemented')

    def __getitem__(self, item):
        raise NotImplementedError('__getitem__ is not implemented')

    @staticmethod
    def _to_tensor(value, d_type=torch.int64):
        return torch.tensor(value, device=CONFIG.DEVICE, dtype=d_type)
