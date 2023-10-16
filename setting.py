import os

import torch


class Config:
    ROOT = '/Users/changyeol/Project/personal_repo/recomm-sisa-pytorch'
    DATASET = os.path.join(ROOT, 'datasets/movielens/ml-1m/ratings.dat')
    TRAIN_DATA = os.path.join(ROOT, 'datasets/movielens/train.csv')
    VAL_DATA = os.path.join(ROOT, 'datasets/movielens/val.csv')
    TEST_DATA = os.path.join(ROOT, 'datasets/movielens/test.csv')
    MAPPER = os.path.join(ROOT, 'datasets/movielens/mapper.json')
    DEVICE = torch.device('cpu')

    PAD_ID = 0
