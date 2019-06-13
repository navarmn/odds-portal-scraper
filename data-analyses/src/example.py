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

DATAFOLDER = join('..', 'data')

data = pd.read_csv(join(DATAFOLDER, 'matches-Brazil-2004-2017.csv'))
print(data.head())


#
teams1 = data['team1'].unique()
teams2 = data['team2'].unique()
all_teams = np.append(teams1, teams2)
all_teams = np.unique(all_teams)

# tf_01 = TeamNameEncoder()
# tf_01.fit(all_teams)
# print(tf_01.transform(data['team1']))

# #
# tf_02 = GetOdds()
# print(tf_02.fit_transform(data['team1_odds']))

# Create pípelines

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

print('result of Numerical pipeline: \n {}'.format(num_pipeline.fit_transform(data)))

print('result of Categorical pipeline: \n {}'.format(cat_pipeline.fit_transform(data)))


full_pipeline = FeatureUnion(
    transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ]
)

print('result of Full pipeline: \n {}'.format(full_pipeline.fit_transform(data)))

full_pipeline.fit(data)

print('result of Full pipeline 2: \n {}'.format(full_pipeline.transform(data)))



















