from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather data

boston_dataset = load_boston()

data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], 1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

# Take property stats as the mean of existing dataset to get the average value of houses
CRIM = 0
ZN = 1
CHAS = 2
NOX = 3
RM = 4
DIS = 5
RAD = 6
TAX = 7
PTRATIO = 8
LSTAT = 10

property_stats = features.mean().values.reshape(1, 11)

reg = LinearRegression()
reg.fit(features, target)

fitted_vals = reg.predict(features)
mse = mean_squared_error(target, fitted_vals)
rmse = np.sqrt(mse)


def get_price_estimate(nr_rooms,
                       students_per_classroom,
                       distance_from,
                       next_to_river=False,
                       high_confidence=True):
    """Estimate house prices in boston with the help of following params:-
    _______________________________________________________________________

    Keyword arguements:
    nr_rooms: Number of rooms
    students_per_classroom: Houses near schools where the ratio of teacher with student
    next_to_river: Houses next to Charles river
    high_confidence: If True, give variance of 68% and 95% if False"""

    if (nr_rooms < 1) or (students_per_classroom < 0):
        print('Unable to estimate with unrealistic values')
        return

    # Configure Property
    property_stats[0][RM] = nr_rooms
    property_stats[0][PTRATIO] = students_per_classroom
    property_stats[0][DIS] = distance_from / 1000

    if next_to_river:
        property_stats[0][CHAS] = 1
    else:
        property_stats[0][CHAS] = 0

    # Estimate preditions
    log_estimate = reg.predict(property_stats)[0][0]

    if high_confidence:
        upper_bound = log_estimate + 2 * rmse
        lower_bound = log_estimate - 2 * rmse
        interval = 95
    else:
        upper_bound = log_estimate + rmse
        lower_bound = log_estimate - rmse
        interval = 68

    # Convert the values in todays values
    TODAYS_MEDIAN_PRICE = 583.3
    SCALE_FACTOR = TODAYS_MEDIAN_PRICE / np.median(boston_dataset.target)

    hi = round(np.e ** upper_bound * 1000 * SCALE_FACTOR, -2)
    low = round(np.e ** lower_bound * 1000 * SCALE_FACTOR, -2)
    act = round(np.e ** log_estimate * 1000 * SCALE_FACTOR, -2)

    print('Upper Bound in dollar is : $', hi)
    print('Actual estimation in dollar is : $', act)
    print('Lower Bound in dollar is : $', low)
    print(f'Interval is {interval}')
