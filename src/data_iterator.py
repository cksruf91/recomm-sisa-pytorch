import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from setting import Config

CONFIG = Config()


class DataIterator(Dataset):

    def __init__(self, dataset: pd.DataFrame):
        self._parsing_dataset(dataset)

    def __len__(self):
        return self._length // 10

    def _parsing_dataset(self, dataset: pd.DataFrame):
        dataset['SessionID'] = dataset['SessionID'].astype(int)
        dataset['Target'] = dataset['Target'].astype(int)
        dataset['Session'] = dataset['Session'].map(eval)
        dataset['Negative'] = dataset['Negative'].map(eval)

        self.dataset = []
        for row in dataset.to_dict(orient='records'):
            session = self._to_tensor(row['Session'])
            padding_mask = torch.where(session == CONFIG.PAD_ID, 1., 0.)
            negative = self._to_tensor(row['Negative'])
            target = self._to_tensor(row['Target'])
            self.dataset.append(
                (session, padding_mask, negative, target)
            )
        self._length: int = len(self.dataset)

    def __getitem__(self, item):
        session, padding_mask, negative, target = self.dataset[item]
        if np.random.random() > 0.5:
            return session, padding_mask, target, self._to_tensor([1.], d_type=torch.float)
        else:
            idx = torch.randint(0, negative.size(0), (1,))
            return session, padding_mask, negative[idx].squeeze(0), self._to_tensor([0.], d_type=torch.float)

    @staticmethod
    def _to_tensor(value, d_type=torch.int64):
        return torch.tensor(value, device=CONFIG.DEVICE, dtype=d_type)
