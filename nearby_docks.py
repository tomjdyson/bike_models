import numpy as np
import pandas as pd
from get_data import pull_remove_na
from poisson import rate_of_success
from sklearn.linear_model import LinearRegression


# data = pull_remove_na()
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# non_corona = data[data['timestamp'] < pd.to_datetime('2020/03/01')]
#
# dock = '1'
#
# other_docks = non_corona.drop([dock, 'timestamp'], axis=1)
# rates, keep_df = rate_of_success(non_corona, dock, timeframe=15)
# rate_df = keep_df.merge(rates.reset_index(), left_on=['weekday', 'hour', 'minute'],
#                         right_on=['weekday', 'hour', 'minute'])
#
# full_array = other_docks.values
# empty_binary_array = np.zeros(full_array.shape)
# empty_binary_array[full_array == 0] = 1
# full_dock_array = full_array / full_array.max(axis=0)
# full_binary_array = np.zeros(full_array.shape)
# full_binary_array[full_dock_array == 1] = 1
#
# empty_clf = LinearRegression()
# empty_clf.fit(empty_binary_array, (rate_df['change_x'] - rate_df['change_y']).fillna(0))
# full_clf = LinearRegression()
# full_clf.fit(full_binary_array, (rate_df['change_x'] - rate_df['change_y']).fillna(0))
#
# pd.DataFrame({'empty_coef': empty_clf.coef_, 'full_coef': full_clf.coef_,
#               'empty_count': empty_binary_array.sum(axis=0),
#               'full_count': full_binary_array.sum(axis=0)}, index=list(other_docks)).to_csv('dock_coef.csv')


class NearbyDockImpact:
    def __init__(self, dock):
        self.dock = dock
        self.clf = LinearRegression
        self.full_value = None
        self.empty_clf = self.clf(fit_intercept=False)
        self.full_clf = self.clf(fit_intercept=False)

    def create_binary_array(self, df, find_value):
        df_array = df.values
        binary_array = np.zeros(df_array.shape)
        binary_array[df_array == find_value] = 1
        return binary_array

    def fit(self, df, rate_df):
        other_docks = df.drop([self.dock, 'timestamp'], axis=1)
        rate_df['adjusted_rate'] = (rate_df['bike_change'] - rate_df['bike_change_average']).fillna(0)
        self.full_value = df[self.dock].max()
        empty_binary_array = self.create_binary_array(other_docks, 0)
        full_binary_array = self.create_binary_array(other_docks, self.full_value)
        self.empty_clf.fit(empty_binary_array, rate_df['adjusted_rate'])
        self.full_clf.fit(full_binary_array, rate_df['adjusted_rate'])
        self.empty_clf.coef_[self.empty_clf.coef_ < 0] = 0
        self.empty_clf.coef_[self.empty_clf.coef_ > 10] = 0
        self.full_clf.coef_[self.full_clf.coef_ > 0] = 0
        self.full_clf.coef_[self.full_clf.coef_ < -10] = 0
        return self

    def predict(self, df):
        other_docks = df.drop([self.dock, 'timestamp'], axis=1)
        empty_binary_array = self.create_binary_array(other_docks, 0)
        full_binary_array = self.create_binary_array(other_docks, self.full_value)
        empty_rates = self.empty_clf.predict(empty_binary_array)
        full_rates = self.full_clf.predict(full_binary_array)
        return empty_rates + full_rates


if __name__ == '__main__':
    from transform_data import DockTransform, BikesTransform

    data = pull_remove_na()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    available_docks = list(data.drop('timestamp', axis=1))
    non_corona = data[data['timestamp'] < pd.to_datetime('2020/03/01')]
    example_df = non_corona[non_corona['timestamp'] > pd.to_datetime('2020/01/01')]
    bike_transformer = BikesTransform()
    bike_transformed_df = bike_transformer.transform(example_df)
    models = []
    dock = '1'
    dock_transformer = DockTransform(dock)
    dock_df = bike_transformed_df[bike_transformer.added_cols + [dock]]
    rates = dock_transformer.transform(dock_df)
    rate_df = dock_transformer.rate_df(dock_df, rates)
    nearby = NearbyDockImpact(dock)
    nearby.fit(example_df, rate_df)

    test = data[
        (data['timestamp'] < pd.to_datetime('2020/03/08')) & (data['timestamp'] >= pd.to_datetime('2020/03/01'))]
    bike_test_df = bike_transformer.transform(test)
    a, b = nearby.predict(test)
    print(a, b)
