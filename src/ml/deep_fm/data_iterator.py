from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from setting import Config
from src.ml.common.iterator import DataIterator

CONFIG = Config()


class DeepFMIterator(DataIterator):

    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)

    def _parsing_row(self, row) -> Tuple[Tensor, Tensor, Tensor]:
        user_id = self._to_tensor([row['UserID']])
        session = self._to_tensor(row['Session'] + [row['Target']])
        negative = self._to_tensor(row['Negative'])
        return user_id, session, negative

    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        user_id, session, negative = self.dataset[item]
        if np.random.random() > 0.5:
            idx = torch.randint(0, session.size(0), (1,))
            x = torch.concat([user_id, session[idx]], dim=-1)
            label = self._to_tensor([1.], d_type=torch.float)
        else:
            idx = torch.randint(0, negative.size(0), (1,))
            x = torch.concat([user_id, negative[idx]], dim=-1)
            label = self._to_tensor([0.], d_type=torch.float)
        return x, label
