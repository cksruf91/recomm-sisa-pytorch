import json
import argparse

import pandas as pd

from src.trainer import ModelTrainer
from src.preprocess import PreProcess
from src.ml.deep_fm.data_iterator import DeepFMIterator
from src.ml.deep_fm.model import DeepFactorizationMachineModel
from src.ml.sisa.data_iterator import SisaIterator
from src.ml.sisa.model import SessionInterestSelfAttention
from setting import Config

CONFIG = Config()


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--run', required=True, choices=['preprocess', 'sisa', 'deepfm'], type=str
    )
    return parser.parse_args()


def main():
    args = get_argument()
    if args.run == 'preprocess':
        PreProcess().run().save()
    elif args.run == 'sisa':
        mapper = json.load(open(CONFIG.MAPPER, mode='r'))
        max_item_id = int(max(mapper.values()))

        train_data = SisaIterator(pd.read_csv(CONFIG.TRAIN_DATA))
        val_data = SisaIterator(pd.read_csv(CONFIG.VAL_DATA))
        test_data = SisaIterator(pd.read_csv(CONFIG.TEST_DATA))
        model = SessionInterestSelfAttention(num_item=max_item_id + 1, emb_dim=64, max_len=20, pad_id=CONFIG.PAD_ID)
        ModelTrainer(arch='sisa', model=model, epoch=100, learning_rate=0.01)\
            .run(train_data=train_data, val_data=val_data, early_stop=7, monitor='val_acc')\
            .test(test_data=test_data)

    elif args.run == 'deepfm':
        mapper = json.load(open(CONFIG.MAPPER, mode='r'))
        max_item_id = int(max(mapper.values()))
        train_data = pd.read_csv(CONFIG.TRAIN_DATA)
        max_user_id = train_data['UserID'].max()

        train_data = DeepFMIterator(train_data)
        val_data = DeepFMIterator(pd.read_csv(CONFIG.VAL_DATA))
        test_data = DeepFMIterator(pd.read_csv(CONFIG.TEST_DATA))
        model = DeepFactorizationMachineModel(
            field_dims=[max_user_id, max_item_id], embed_dim=64, mlp_dims=[64, 32], dropout=0.2, device=CONFIG.DEVICE
        )
        ModelTrainer(arch='deepfm', model=model, epoch=100, learning_rate=0.01)\
            .run(train_data=train_data, val_data=val_data, early_stop=7, monitor='val_acc')\
            .test(test_data=test_data)


if __name__ == '__main__':
    main()
