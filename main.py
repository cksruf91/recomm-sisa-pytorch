import json
import argparse

import pandas as pd

from src.train import Train
from src.preprocess import PreProcess
from src.data_iterator import DataIterator
from src.ml.model.sisa import SessionInterestSelfAttention
from setting import Config

CONFIG = Config()


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--type', required=True, choices=['data', 'train'], type=str
    )
    return parser.parse_args()


def main():
    args = get_argument()
    if args.type == 'data':
        PreProcess().run().save()
    elif args.type == 'train':
        mapper = json.load(open(CONFIG.MAPPER, mode='r'))
        max_item_id = int(max(mapper.values()))

        train_data = DataIterator(pd.read_csv(CONFIG.TRAIN_DATA))
        val_data = DataIterator(pd.read_csv(CONFIG.VAL_DATA))
        model = SessionInterestSelfAttention(num_item=max_item_id + 1, emb_dim=128, max_len=20, pad_id=CONFIG.PAD_ID)
        Train(model=model, epoch=50).run(train_data=train_data, val_data=val_data)


if __name__ == '__main__':
    main()
