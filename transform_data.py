class BikesTransform:
    def __init__(self):
        self.added_cols = []

    def add_cols(self, df):
        df.loc[:, 'weekday'] = 1
        df.loc[df['timestamp'].dt.weekday.isin([5, 6]), 'weekday'] = 0
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        self.added_cols += ['weekday', 'hour', 'minute']
        return df

    def transform(self, df):
        new_df = df.copy()
        return self.add_cols(new_df)


class DockTransform:
    def __init__(self, dock, time_step=15):
        self.time_step = time_step
        self.group_cols = ['weekday', 'hour'] if self.time_step == 60 else ['weekday', 'hour', 'minute']
        self.dock = dock

    def add_cols(self, df, dock):
        df['empty'] = 0
        df['full'] = 0
        df.loc[df[dock] == 0, 'empty'] = 1
        df.loc[df[dock] == df[dock].max(), 'full'] = 1
        return df

    def add_change(self, df, dock):
        df['bike_change'] = -df[dock].diff(periods=-1)
        return df

    def find_time_rates(self, df):
        net_rate = df[(df['empty'] == 0) & (df['full'] == 0)].groupby(self.group_cols)['bike_change'].mean()
        return net_rate

    def transform(self, df):
        new_df = df.copy()
        with_cols = self.add_cols(new_df, self.dock)
        with_change = self.add_change(with_cols, self.dock)
        rates = self.find_time_rates(with_change)
        return rates

    def rate_df(self, df, rates=None):
        with_cols = self.add_cols(df, self.dock)
        with_change = self.add_change(with_cols, self.dock)
        if rates is None:
            rates = self.find_time_rates(with_change)
        rate_df = with_change.merge(rates.reset_index(), left_on=['weekday', 'hour', 'minute'],
                                    right_on=['weekday', 'hour', 'minute'], suffixes=['', '_average'])
        return rate_df
