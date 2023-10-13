from typing import Optional, List, Set

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from setting import Config

CONFIG = Config()


class PreProcess:

    def __init__(self):
        print('PreProcess run')
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def run(self) -> Self:
        ratings = self._data_load()

        print('Preprocess dataset')
        ratings.groupby('UserID')['MovieID'].count()

        # SessionID 생성
        ratings.sort_values(['UserID', 'Timestamp'], inplace=True)
        ratings['diff'] = ratings.groupby('UserID')['Timestamp'].diff()
        ratings.loc[(ratings['diff'] > 60 * 30) | ratings['diff'].isnull(), 'SessionID'] = 1
        ratings['SessionID'] = ratings['SessionID'].fillna(0).cumsum()
        ratings['SessionID'] -= 1

        # 10건 이하 세션 제거
        session_count = ratings.groupby('SessionID').size()
        ratings = ratings.merge(session_count[session_count > 10].reset_index(), on='SessionID', how='inner')

        train, test = train_test_split(ratings, test_size=0.2)
        val, test = train_test_split(test, test_size=0.5)

        # ItemID 생성
        drop_cols = set(ratings.columns) - {'SessionID', 'ItemID'}

        mapper = {movie_id: item_id + 1 for item_id, movie_id in enumerate(train['MovieID'].unique())}
        train['ItemID'] = train['MovieID'].map(lambda x: mapper[x])
        train.drop(columns=drop_cols, inplace=True)

        val['ItemID'] = val['MovieID'].map(lambda x: mapper.get(x))
        drop_index = val[val['ItemID'].isnull()].index
        val.drop(index=drop_index, columns=drop_cols, inplace=True)

        test['ItemID'] = test['MovieID'].map(lambda x: mapper.get(x))
        drop_index = test[test['ItemID'].isnull()].index
        test.drop(index=drop_index, columns=drop_cols, inplace=True)

        pop_item = set(train.groupby('ItemID')
                       .size().sort_values(ascending=False)
                       .head(1000).index)

        self.train_data = self._aggregate(train, pop_item=pop_item)
        self.val_data = self._aggregate(val, pop_item=pop_item)
        self.test_data = self._aggregate(test, pop_item=pop_item)

        return self

    def save(self):
        print('save dataset')
        self.train_data.to_csv(CONFIG.TRAIN_DATA, index=True)
        self.val_data.to_csv(CONFIG.VAL_DATA, index=True)
        self.test_data.to_csv(CONFIG.TEST_DATA, index=True)

    def _aggregate(self, dataset, pop_item):
        return dataset.groupby('SessionID').agg(
            Session=pd.NamedAgg(column='ItemID', aggfunc=lambda x: self._padding(list(x)[-21:-1], length=20)),
            Target=pd.NamedAgg(column='ItemID', aggfunc=lambda x: list(x)[-1]),
            Negative=pd.NamedAgg(column='ItemID', aggfunc=lambda x: self._negative_sampling(pop_item, x, k=50))
        )

    @staticmethod
    def _data_load() -> pd.DataFrame:
        print('loading dataset')
        ratings_header = "UserID::MovieID::Rating::Timestamp"
        ratings = pd.read_csv(
            CONFIG.DATASET, sep='::', header=None, names=ratings_header.split('::'), engine='python'
        )
        return ratings

    @staticmethod
    def _padding(seq: List[int], length: int) -> List[int]:
        pad_len = length - len(seq)
        seq = seq + [CONFIG.PAD_ID for _ in range(pad_len)]
        return seq

    @staticmethod
    def _negative_sampling(item_pool: Set, seq: List, k: int = 50) -> List[int]:
        sample = list(item_pool - set(seq))
        return list(np.random.choice(sample, size=k, replace=False))
