import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
import time
SEED = int(time.time())

import sys
sys.path.append('..')

from os.path import join
from datetime import datetime

from transforms import *
from model_selection import *

DATAFOLDER = join('..', 'data')

data = pd.read_csv(join(DATAFOLDER, 'matches-Brazil-2004-2017.csv'))
print(data.head())


# Create pipelines

num_attribs = ['team1_odds', 'team2_odds', 'draw_odds']
cat_attribs = ['team1', 'team2']

num_pipeline = Pipeline(
    [
        ('selector', DataFrameSelector(attribute_names=num_attribs)),
        ('odds2float', GetOdds()),
        ('std_scaller', StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    [
        ('selector', DataFrameSelector(attribute_names=cat_attribs)),
        ('encoder', TeamNameEncoder())
    ]
)

full_pipeline = FeatureUnion(
    transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ]
)

# Model selection

kf = DateKFold()
ui = []
for train_idx, test_idx in kf.split(data):
    print(data.iloc[test_idx])
    print("TRAIN: \n {}".format(full_pipeline.fit_transform(data.iloc[train_idx])))
    print("TESTE: \n {}".format(full_pipeline.transform(data.iloc[test_idx])))

















