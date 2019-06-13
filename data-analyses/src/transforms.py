import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler   
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class TeamNameEncoder(TransformerMixin, BaseEstimator):
    '''
    Nothing yet.
    '''
    def __init__(self, type='label'):
        self.type = type
        if self.type == '1hot':
            self.encoder = OneHotEncoder(handle_unknown='ignore')
        elif self.type == 'label':
            self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        '''
        teams_names: array
        '''
        if self.type == '1hot':
            teams1 = X[:,0]
            teams2 = X[:,1]
            all_teams = np.append(teams1, teams2)
            all_teams = np.unique(all_teams)
            self.encoder.fit(all_teams.reshape(-1,1))  
            return self

        elif self.type == 'label':
            teams1 = X[:,0]
            teams2 = X[:,1]
            all_teams = np.append(teams1, teams2)
            all_teams = np.unique(all_teams)
            self.encoder.fit(all_teams)     
            return self

    def transform(self, X, y=None):
        if self.type == '1hot':
            n_teams = len(self.encoder.categories_[0])
            output = np.zeros((X.shape[0], n_teams))
            for col in range(0, X.shape[1]):
                if col is 0:
                    result = self.encoder.transform(X[:,col].reshape(-1,1)).toarray()
                    output[:, :n_teams] = result
                else:
                    result = self.encoder.transform(X[:,col].reshape(-1,1)).toarray()
                    np.concatenate((output, result), axis=1)
            return output
        elif self.type == 'label':
            output = X.copy()
            for col in range(0, output.shape[1]):
                output[:,col] = self.encoder.transform(output[:,col])
            return output

class GetOdds(TransformerMixin, BaseEstimator):
    '''
    Nothing yet.
    '''
    def _odd2float(self, s):
        try:
            return float(s)
        except:
            return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        func = np.vectorize(self._odd2float)
        return func(X)


        