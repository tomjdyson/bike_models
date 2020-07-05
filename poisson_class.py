import numpy as np


class PoissonModel:

    def __init__(self, dock, max_bikes, max_future_steps=4, nearby_dock=None):
        self.max_future_steps = max_future_steps
        self.rate_arrays = {}
        self.dock = dock
        self.max_bikes = max_bikes
        self.h_r_map = {}
        self.nearby_dock = nearby_dock

    def rate_array(self, rates, weekday):
        rates_df = rates.reset_index()
        weekday_array = rates_df.loc[rates_df['weekday'] == weekday, 'bike_change'].values
        future_arrays = []
        for i in range(self.max_future_steps):
            future_arrays.append(np.roll(weekday_array, -i))
        future_array = np.cumsum(np.stack(future_arrays), axis=0)
        return future_array

    def combine_hour_minute(self, df):
        df['combined_h_r'] = df['hour'].astype(str) + '_' + df['minute'].astype(str)
        return df

    def create_h_r_map(self, df):
        df = df[df['weekday'] == 0]
        self.h_r_map = {value: key for key, value in df['combined_h_r'].to_dict().items()}

        if len(self.h_r_map) != 96:
            raise ValueError(f'rate does not contain all timestamps, needs 96 but has {len(self.h_r_map)}')

    def create_rate_array(self, rates):
        rate_df = rates.reset_index()
        rate_df = self.combine_hour_minute(rate_df)
        self.create_h_r_map(rate_df)
        for i in [0, 1]:
            self.rate_arrays[i] = self.rate_array(rates, i)

    def fit(self, rates):
        self.create_rate_array(rates)
        return self

    def other_impact(self, df):
        if self.nearby_dock is not None:
            adjust_rates = self.nearby_dock.predict(
                df.drop(['weekday', 'hour', 'minute', 'combined_h_r', 'time_map', 'rate'], axis=1))
        else:
            adjust_rates = [0 for i in range(len(df))]
        return adjust_rates

    def factorial(self, n):
        fact = 1
        for i in range(1, int(n) + 1):
            fact = fact * i
        return fact

    def poisson_prob(self, rate, num_bikes):
        return (rate ** num_bikes * np.exp(-rate)) / self.factorial(num_bikes)

    def cumulative_poisson(self, rate, k):
        cumulative = 0.0
        i = 0
        while i < k:
            cumulative += (rate ** i) / self.factorial(i)
            i += 1
        cumulative *= np.exp(-rate)
        return 1 - cumulative

    def empty_probability(self, rate, num_bikes):
        if (rate >= 0) & (num_bikes > 0):
            return 0
        elif (rate >= 0) & (num_bikes == 0):
            return 1 - self.cumulative_poisson(rate, 1)
        else:
            return self.cumulative_poisson(-rate, num_bikes)

    def full_probability(self, rate, num_bikes):
        if rate >= 0:
            return self.cumulative_poisson(rate, self.max_bikes - num_bikes)
        else:
            return 0

    def predict_proba(self, df, ahead=1):
        new_df = df.copy()
        new_df = self.combine_hour_minute(new_df)
        new_df['time_map'] = new_df['combined_h_r'].map(self.h_r_map)
        new_df['rate'] = new_df[['time_map', 'weekday']].apply(lambda x: self.rate_arrays[x.weekday][ahead - 1, x.time_map],
                                                       axis=1)
        new_df['nearby_adjusted_rate'] = new_df['rate'] + self.other_impact(new_df)
        new_df['empty_probability'] = new_df[['nearby_adjusted_rate', self.dock]].apply(
            lambda x: self.empty_probability(x.nearby_adjusted_rate, x[self.dock]), axis=1)
        new_df['full_probability'] = new_df[['nearby_adjusted_rate', self.dock]].apply(
            lambda x: self.full_probability(x.nearby_adjusted_rate, x[self.dock]), axis=1)

        return new_df['empty_probability'].values, new_df['full_probability'].values

    def predict(self, df, ahead=1):
        empty, full = self.predict_proba(df, ahead)
        empty_zero = np.zeros(empty.shape)
        full_zero = np.zeros(full.shape)
        empty_zero[empty >= 0.8] = 1
        full_zero[full >= 0.8] = 1
        return empty_zero, full_zero
