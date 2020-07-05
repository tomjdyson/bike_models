import pandas as pd
from sqlalchemy import create_engine
import os
import numpy as np
from config import username, password


def pull_all(db='bikes_present'):
    try:
        df = pd.read_csv('data/full_bikes.csv', index_col=0)
    except Exception as e:
        print(e)
        path = os.path.join('mysql+pymysql://' + username + ':' + password +
                            '@bikes-db-1.cxd403f5i8vi.eu-west-2.rds.amazonaws.com:3306/bikes')
        db_engine = create_engine(path)

        df = pd.read_sql(f"""SELECT * FROM bikes.{db}""", db_engine)
        df.to_csv('data/full_bikes.csv')
    return df


def pull_remove_na(db='bikes_present', na_thresh=50):
    df = pull_all(db)
    ts_series = df['timestamp']
    data = df.drop('timestamp', axis=1)

    data = data.dropna(axis=0, how='all')
    data = data.dropna(thresh=(len(data) - 10), axis=1)
    data = data.dropna(axis=0)
    data['timestamp'] = ts_series
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data


def convert_to_binary_empty(df):
    new_df = df.drop('timestamp', axis=1)
    binary_df = new_df.copy()
    binary_df[df == 0] = 1
    binary_df[df != 0] = 0
    return binary_df


def previous_times(x, y):
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    x_list = []
    y_list = []
    for index in range(5, len(x)):
        x_list.append(x.loc[index - 5:index, :].values.copy())
        y_list.append(y.loc[index, :].values.copy())
    return np.rollaxis(np.dstack(x_list), -1), np.vstack(y_list)
