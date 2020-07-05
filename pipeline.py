from transform_data import DockTransform, BikesTransform
from poisson_class import PoissonModel
from get_data import pull_remove_na
import pandas as pd
import numpy as np
from nearby_docks import NearbyDockImpact

data = pull_remove_na()
data['timestamp'] = pd.to_datetime(data['timestamp'])
available_docks = list(data.drop('timestamp', axis=1))
non_corona = data[data['timestamp'] < pd.to_datetime('2020/03/01')]
# 0.9837
example_df = non_corona[non_corona['timestamp'] > pd.to_datetime('2019/12/01')]
bike_transformer = BikesTransform()
bike_transformed_df = bike_transformer.transform(example_df)
models = []
for dock in available_docks:
    dock_transformer = DockTransform(dock)
    dock_df = bike_transformed_df[bike_transformer.added_cols + [dock]]
    rates = dock_transformer.transform(dock_df)
    rate_df = dock_transformer.rate_df(dock_df, rates)
    #nearby = NearbyDockImpact(dock)
    #nearby.fit(example_df, rate_df)
    p_clf = PoissonModel(dock, bike_transformed_df[dock].max())
    p_clf.fit(rates)
    models.append(p_clf)

test = data[(data['timestamp'] < pd.to_datetime('2020/03/08')) & (data['timestamp'] >= pd.to_datetime('2020/03/01'))]
bike_test_df = bike_transformer.transform(test)
empty_probs = []
full_probs = []
behind = 1
for model in models:
    empty, full = model.predict(bike_test_df, behind)
    empty_probs.append(empty.copy())
    full_probs.append(full.copy())

base_empty = test[available_docks]
base_empty_binary = np.zeros(base_empty.shape)
base_empty_binary[base_empty == 0] = 1

compare_empty = test[available_docks].shift(-behind)
compare_empty_binary = np.zeros(compare_empty.shape)
compare_empty_binary[compare_empty == 0] = 1
a = np.stack(empty_probs).T
b = np.stack(full_probs).T

accuracy = (compare_empty_binary == a).sum() / a.size
base = (compare_empty_binary == base_empty_binary).sum() / a.size
bike_present = np.zeros(a.shape)
# bike_present[a <= 0.8] = 1
# compare_empty_binary[compare_empty >= 1] = 1
#
# score = np.zeros(bike_present.shape)
# score[(bike_present == 1) & (compare_empty_binary == 1)] = 1
# score[(bike_present == 1) & (compare_empty_binary == 0)] = -4
# score[(bike_present == 0) & (compare_empty_binary == 0)] = 1
# score[(bike_present == 0) & (compare_empty_binary == 1)] = -1 / 4
# average_score = score[:-behind].mean()
# print(p_clf)
