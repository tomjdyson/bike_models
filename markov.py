import numpy as np
from poisson import approximate_rates


def create_transition_matrix(arrival_rate, departure_rate, capacity):
    capacity = int(capacity)
    tn_matrix = np.zeros((capacity, capacity))
    for i in range(capacity):
        if i == 0:
            tn_matrix[0, 0] = -departure_rate
            tn_matrix[0, 1] = departure_rate
        elif i == (capacity - 1):
            tn_matrix[i, (capacity - 2)] = arrival_rate
            tn_matrix[i, (capacity - 1)] = -arrival_rate
        else:
            tn_matrix[i, i - 1] = arrival_rate
            tn_matrix[i, i] = -(departure_rate + arrival_rate)
            tn_matrix[i, i + 1] = departure_rate
    return tn_matrix


def create_time_arrays(df, dock):
    arrival_rate, departure_rate = approximate_rates(df, dock)
    rates = pd.DataFrame({'arrival_rate': arrival_rate, 'departure_rate': departure_rate}).reset_index()
    time_array = {}
    for index, row in rates.iterrows():
        if str(row.weekday) not in time_array.keys():
            time_array[str(row.weekday)] = {}
        time_array[str(row.weekday)][str(row.hour)] = create_transition_matrix(row.arrival_rate, row.departure_rate,
                                                                max(df[dock])).copy()
    return time_array


if __name__ == '__main__':
    from get_data import pull_remove_na
    import pandas as pd
    from scipy.linalg import expm
    from numpy.random import poisson

    data = pull_remove_na()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    non_corona = data[data['timestamp'] < pd.to_datetime('2020/03/01')]
    example_df = non_corona[non_corona['timestamp'] > pd.to_datetime('2019/10/01')]
    time_array = create_time_arrays(example_df, '11')
    a = expm((time_array['1.0']['8.0'] - time_array['1.0']['7.0']))
    b = expm((time_array['1.0']['1.0'] - time_array['1.0']['0.0']))
    c = expm((time_array['1.0']['20.0'] - time_array['1.0']['17.0']))
    print(a)