from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from setting import Config
from src.ml.common.iterator import DataIterator

CONFIG = Config()


class SisaIterator(DataIterator):

    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)

    def _parsing_row(self, row) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        session = self._to_tensor(row['Session'])
        padding_mask = torch.where(session == CONFIG.PAD_ID, 1., 0.)
        negative = self._to_tensor(row['Negative'])
        target = self._to_tensor(row['Target'])
        return session, padding_mask, negative, target

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        session, padding_mask, negative, target = self.dataset[item]
        if np.random.random() > 0.5:
            return session, padding_mask, target, self._to_tensor([1.], d_type=torch.float)
        else:
            idx = torch.randint(0, negative.size(0), (1,))
            return session, padding_mask, negative[idx].squeeze(0), self._to_tensor([0.], d_type=torch.float)
