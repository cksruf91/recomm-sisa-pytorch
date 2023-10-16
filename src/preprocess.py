import json
from typing import Optional, List, Set, Dict

import numpy as np
import pandas as pd
from typing_extensions import Self

from setting import Config

CONFIG = Config()


class PreProcess:

    def __init__(self):
        print('PreProcess run')
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.item_mapper: Dict[int, int] = {}
        self.user_mapper: Dict[int, int] = {}

    def run(self) -> Self:
        ratings = self._data_load()

        print('Preprocess dataset')

        # SessionID 생성
        ratings.sort_values(['UserID', 'Timestamp'], inplace=True)
        ratings['diff'] = ratings.groupby('UserID')['Timestamp'].diff()
        ratings.loc[(ratings['diff'] > 60 * 10) | ratings['diff'].isnull(), 'SessionID'] = 1
        ratings['SessionID'] = ratings['SessionID'].fillna(0).cumsum()
        ratings['SessionID'] -= 1

        # 길이 5 이하 세션 제거
        _before_shape = ratings.shape
        session_length = ratings.groupby('SessionID').size()
        ratings = ratings.merge(session_length[session_length > 5].reset_index(), on='SessionID', how='inner')
        print(f'session length drop: {_before_shape} -> {ratings.shape}')

        # 세션 3개 이하 유저 제거
        _before_shape = ratings.shape
        session_count = ratings.groupby('UserID').agg(
            NumSession=pd.NamedAgg(column='SessionID', aggfunc='nunique'),
        )
        ratings = ratings.merge(
            session_count[session_count['NumSession'] > 3].reset_index(),
            on='UserID', how='inner'
        )
        print(f'session count drop: {_before_shape} -> {ratings.shape}')

        ratings = self._train_test_val_split(ratings)
        print(ratings['Case'].value_counts(dropna=False))

        # ItemID 생성
        drop_cols = set(ratings.columns) - {'UserID', 'SessionID', 'ItemID', 'Case'}

        movie_ids = ratings.loc[ratings['Case'] == 'train', 'MovieID'].unique()
        self.item_mapper = {int(movie_id): int(item_id + 1) for item_id, movie_id in enumerate(movie_ids)}
        ratings['ItemID'] = ratings['MovieID'].map(lambda x: self.item_mapper.get(x))

        _before_shape = ratings.shape
        drop_index = ratings[ratings['ItemID'].isnull()].index
        ratings.drop(index=drop_index, columns=drop_cols, inplace=True)

        print(f'create item id: {_before_shape} -> {ratings.shape}')

        # UserID 재생성
        self.user_mapper = {int(_user_id): int(user_id) for user_id, _user_id in enumerate(ratings['UserID'])}
        ratings['UserID'] = ratings['UserID'].map(lambda x: self.user_mapper[x])

        pop_item = set(ratings[ratings['Case'] == 'train'].groupby('ItemID')
                       .size().sort_values(ascending=False)
                       .head(1000).index)

        self.train_data = self._aggregate(ratings[ratings['Case'] == 'train'], pop_item=pop_item)
        self.val_data = self._aggregate(ratings[ratings['Case'] == 'val'], pop_item=pop_item)
        self.test_data = self._aggregate(ratings[ratings['Case'] == 'test'], pop_item=pop_item)

        return self

    def save(self):
        print('save dataset')
        self.train_data.to_csv(CONFIG.TRAIN_DATA, index=True)
        self.val_data.to_csv(CONFIG.VAL_DATA, index=True)
        self.test_data.to_csv(CONFIG.TEST_DATA, index=True)
        json.dump(self.item_mapper, open(CONFIG.MAPPER, mode='w'))

    def _aggregate(self, dataset, pop_item):
        return dataset.groupby('SessionID').agg(
            UserID=pd.NamedAgg(column='UserID', aggfunc=max),
            Session=pd.NamedAgg(column='ItemID', aggfunc=lambda x: self._padding(list(x)[-21:-1], length=20)),
            Target=pd.NamedAgg(column='ItemID', aggfunc=lambda x: list(x)[-1]),
            Negative=pd.NamedAgg(column='ItemID', aggfunc=lambda x: self._negative_sampling(pop_item, x, k=200))
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

    @staticmethod
    def _train_test_val_split(df: pd.DataFrame):
        session = df.groupby('UserID').agg(
            SessionID=pd.NamedAgg(column='SessionID', aggfunc='max'),
            Test=pd.NamedAgg(column='SessionID', aggfunc=lambda x: 1),
            Val=pd.NamedAgg(column='SessionID', aggfunc=lambda x: 1),
        ).reset_index()

        df = df.merge(session[['UserID', 'SessionID', 'Test']],
                      on=['UserID', 'SessionID'], how='left', validate='m:1')

        session['SessionID'] -= 1
        df = df.merge(session[['UserID', 'SessionID', 'Val']],
                      on=['UserID', 'SessionID'], how='left', validate='m:1')

        df.loc[df['Test'] == 1, 'Case'] = 'test'
        df.loc[df['Val'] == 1, 'Case'] = 'val'
        df['Case'].fillna('train', inplace=True)
        return df.drop(columns=['Test', 'Val'], inplace=False)
