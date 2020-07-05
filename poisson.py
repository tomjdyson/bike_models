import datetime
import numpy as np
import pandas as pd


def factorial(n):
    fact = 1
    for i in range(1, int(n) + 1):
        fact = fact * i
    return fact


def poisson_prob(rate, num_bikes):
    return (rate ** num_bikes * np.exp(-rate)) / factorial(num_bikes)


def cumulative_poisson(rate, k):
    cumulative = 0.0
    i = 0
    while i < k:
        cumulative += (rate ** i) / factorial(i)
        i += 1
    cumulative *= np.exp(-rate)
    return 1 - cumulative


def empty_poisson(rate, num_bikes):
    if rate >= 0:
        return 0
    else:
        return cumulative_poisson(-rate, num_bikes)


def full_poisson(rate, num_bikes, max_bikes):
    if rate >= 0:
        return cumulative_poisson(rate, max_bikes - num_bikes)
    else:
        return 0


def arrival_rates(series):
    bike_change = series.diff(periods=-1)
    arrival = bike_change[bike_change > 0]
    return arrival


def departure_rates(series):
    bike_change = series.diff(periods=-1)
    arrival = bike_change[bike_change < 0]
    return arrival


def find_empty(df, dock):
    df['dow'] = df['timestamp'].dt.weekday
    df['weekday'] = 1
    df.loc[df['dow'].isin([5, 6]), 'weekday'] = 0
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['empty'] = 0
    df['full'] = 0
    df.loc[df[dock] == 0, 'empty'] = 1
    df.loc[df[dock] == df[dock].max(), 'full'] = 1

    df['change'] = -df[dock].diff(periods=-1)
    keep_df = df[['weekday', 'hour', 'minute', 'empty', 'full', dock, 'change']]
    group_cols = ['weekday', 'hour'] if 60 == 60 else ['weekday', 'hour', 'minute']
    net_rate = keep_df.groupby(group_cols)['change'].mean()
    empty_change = keep_df[keep_df['empty'] == 1].groupby(group_cols)['change'].mean()
    empty_count = keep_df[keep_df['empty'] == 1].groupby(group_cols)['change'].count()
    large_sample = empty_change[empty_count > 50]
    net_rate.name = 'net_change'
    a = pd.concat([net_rate, large_sample], axis=1).dropna()
    a['dock'] = dock
    return a.reset_index()


def rate_of_success(df, dock, timeframe=60):
    df['dow'] = df['timestamp'].dt.weekday
    df['weekday'] = 1
    df.loc[df['dow'].isin([5, 6]), 'weekday'] = 0
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['empty'] = 0
    df['full'] = 0
    df.loc[df[dock] == 0, 'empty'] = 1
    df.loc[df[dock] == df[dock].max(), 'full'] = 1

    df['change'] = -df[dock].diff(periods=-1)

    keep_df = df[['weekday', 'hour', 'minute', 'empty', 'full', dock, 'change']]
    group_cols = ['weekday', 'hour'] if timeframe == 60 else ['weekday', 'hour', 'minute']

    net_rate = keep_df[(keep_df['empty'] == 0) & (keep_df['full'] == 0)].groupby(group_cols)['change'].mean()
    return net_rate, keep_df


def approximate_rates(df, dock, timeframe=60):
    df['dow'] = df['timestamp'].dt.weekday
    df['weekday'] = 1
    df.loc[df['dow'].isin([5, 6]), 'weekday'] = 0
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['empty'] = 0
    df['full'] = 0
    df.loc[df[dock] == 0, 'empty'] = 1
    df.loc[df[dock] == df[dock].max(), 'full'] = 1

    df['change'] = -df[dock].diff(periods=-1)

    keep_df = df[['weekday', 'hour', 'minute', 'empty', 'full', dock, 'change']]
    group_cols = ['weekday', 'hour'] if timeframe == 60 else ['weekday', 'hour', 'minute']

    net_rate = keep_df.groupby(group_cols)['change'].mean()

    arrivals_pos = net_rate[net_rate >= 0]
    arrivals_neg = net_rate[net_rate < 0]

    arrival_pos_rates = arrivals_pos * 0.8944 + 0.2304
    arrival_neg_rates = arrivals_neg * -0.3574 + 0.0944
    arrival_rate = pd.concat([arrival_pos_rates, arrival_neg_rates])
    departure_rate = net_rate - arrival_rate

    return arrival_rate, - departure_rate


if __name__ == '__main__':
    from get_data import pull_remove_na
    from markov import create_transition_matrix
    import pandas as pd

    data = pull_remove_na()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    non_corona = data[data['timestamp'] < pd.to_datetime('2020/03/01')]
    example_df = non_corona[non_corona['timestamp'] > pd.to_datetime('2020/01/01')]
    dock = '2'
    rates, keep_df = rate_of_success(example_df, dock, timeframe=15)
    a = keep_df.merge(rates.reset_index(), left_on=['weekday', 'hour', 'minute'],
                      right_on=['weekday', 'hour', 'minute'])
    dock_max = a[dock].max()
    a['empty_probability'] = a.apply(lambda x: empty_poisson(x.change_y, x[dock]), axis=1)
    a['full_probability'] = a.apply(lambda x: full_poisson(x.change_y, x[dock], dock_max), axis=1)
    print(rates)
    # found_dfs = []
    # for dock in list(data.drop('timestamp', axis=1)):
    #     found_dfs.append(find_empty(example_df, dock).copy())
    #
    # all_dfs = pd.concat(found_dfs)
    # all_dfs.to_csv('empty_rates.csv')
    # print(all_dfs)
    # tn_mx = create_transition_matrix(arrive_rate.iloc[0], depart_rate.iloc[0], max(example_df['1']))
    # print(tn_mx)
