import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from sklearn.model_selection import KFold

DATEFIELD = 'start_time'

class DateKFold(object):
    '''
    Nothing yet
    '''

    def __init__(self, date_type='year', field=DATEFIELD):
        self.date_type = date_type
        self.field = field


    def _selector(self, df):
        output = df[DATEFIELD]
        output[self.field] = output[self.field].apply(self._set_datetime_obj) 
        return output


    def _set_datetime_year(self, date_str):
        date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return date.year

    def _create_pars(self, df):
        # df_buffer = self._selector(df)
        df_buffer = df.copy()
        if self.date_type == 'year':
            df_buffer['year'] = df[self.field].apply(self._set_datetime_year) 
            self.indices = df_buffer.groupby('year').indices 
            self.years_range = np.array(list(self.indices.keys()))

            return self

    def split(self, df):
        self._create_pars(df)
        for y in range(1, max(self.years_range.shape)-1):
            years_train = self.years_range[:y+1]
            years_test = self.years_range[y+1:]

            train_idx = np.array([])
            test_idx = np.array([])
            for year in years_train:
                train_idx = np.append(train_idx, self.indices[year])
            for year in years_test:
                test_idx = np.append(test_idx, self.indices[year])

            yield train_idx, test_idx
                






            




        