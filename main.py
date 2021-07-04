import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pickle
import datetime

from darksky.api import DarkSky
from darksky.types import languages, units, weather
# from datetime import datetime, timedelta
from datetime import timedelta
from collections import namedtuple


def extract_weather_data(api_key, target_date, lat, lont, days):
    records = []
    darksky = DarkSky(api_key)
    DailySummary = get_daily_summary()
    for _ in range(days):
        forecast = darksky.get_time_machine_forecast(
            lat, lont,
            target_date,
            extend=False,  # default `False`
            lang=languages.ENGLISH,  # default `ENGLISH`
            units=units.AUTO,  # default `auto`
            exclude=[weather.ALERTS]  # default `[]`
        )

        buf = []
        for i in range(len(forecast.hourly.data)):
            try:
                buf.append(forecast.hourly.data[i].dew_point)
            except:
                pass
        if len(buf) == 0:
            max_dewpt = 0.0
            min_dewpt = 0.0
        else:
            max_dewpt = max(buf)
            min_dewpt = min(buf)

        buf = []
        for i in range(len(forecast.hourly.data)):
            try:
                buf.append(forecast.hourly.data[i].humidity)
            except:
                pass
        if len(buf) == 0:
            max_humidity = 0.0
            min_humidity = 0.0
        else:
            max_humidity = max(buf)
            min_humidity = min(buf)

        buf = []
        for i in range(0, len(forecast.hourly.data), 3):
            try:
                buf.append(forecast.hourly.data[i].pressure)
            except:
                pass

        if len(buf) == 0:
            max_pressure = 0.0
            min_pressure = 0.0
            mean_pressure = 0.0
        else:
            max_pressure = max(buf)
            min_pressure = min(buf)
            mean_pressure = sum(buf) / len(buf)

        try:
            preciep_prob = forecast.daily.data[0].precip_probability
        except:
            preciep_prob = 0.0

        for item in forecast.daily.data:
            records.append(DailySummary(
                date=target_date,
                meantemp=(item.temperature_max + item.temperature_min) / 2,
                maxtemp=item.temperature_max,
                mintemp=item.temperature_min,
                meandewpt=item.dew_point,
                maxdewpt=max_dewpt,
                mindewpt=min_dewpt,
                meanhumidity=item.humidity,
                maxhumidity=max_humidity,
                minhumidity=min_humidity,
                meanpressure=mean_pressure,
                maxpressure=max_pressure,
                minpressure=min_pressure,
                precippr=preciep_prob
            ))
        target_date += timedelta(days=1)
    return records


def derive_nth_day_feature(df, feature, n):
    rows = df.shape[0]
    nth_prior_measurements = [None] * n + [df[feature][i - n] for i in range(n, rows)]
    col_name = "{}_{}".format(feature, n)
    df[col_name] = nth_prior_measurements


def features_list():
    features = ["date", "meantemp", "maxtemp", "mintemp", "meandewpt", "maxdewpt", "mindewpt", "meanhumidity",
                "maxhumidity", "minhumidity", "meanpressure", "maxpressure", "minpressure", "precippr"]
    return features


def get_daily_summary():
    DailySummary = namedtuple("DailySummary", features_list())
    return DailySummary


def clear_data(records):

    features = features_list()
    DailySummary = get_daily_summary()

    df = pd.DataFrame(records, columns=features).set_index('date')

    df.loc[:, 'meanhumidity'] *= 100
    df.loc[:, 'maxhumidity'] *= 100
    df.loc[:, 'minhumidity'] *= 100

    # using dictionary to convert specific columns
    convert_dict = {"meantemp": int,
                    "maxtemp": int,
                    "mintemp": int,
                    "meandewpt": int,
                    "maxdewpt": int,
                    "mindewpt": int,
                    "meanhumidity": int,
                    "maxhumidity": int,
                    "minhumidity": int,
                    "meanpressure": int,
                    "maxpressure": int,
                    "minpressure": int
                    }
    df = df.astype(convert_dict)

    # using dictionary to convert specific columns
    convert_dict = {"meantemp": float,
                    "maxtemp": float,
                    "mintemp": float,
                    "meandewpt": float,
                    "maxdewpt": float,
                    "mindewpt": float,
                    "meanhumidity": float,
                    "maxhumidity": float,
                    "minhumidity": float,
                    "meanpressure": float,
                    "maxpressure": float,
                    "minpressure": float
                    }
    df = df.astype(convert_dict)
    
    for feature in features:
        if feature != 'date':
            for N in range(1, 4):
                derive_nth_day_feature(df, feature, N)

    # make list of original features without meantempm, mintempm, and maxtempm
    to_remove = [feature
                 for feature in features
                 if feature not in ['meantemp', 'mintemp', 'maxtemp', 'meanhumidity', 'meanpressure']]

    # make a list of columns to keep
    to_keep = [col for col in df.columns if col not in to_remove]

    # select only the columns in to_keep and assign to df
    df = df[to_keep]
    df = df.apply(pd.to_numeric, errors='coerce')
    # df.info()

    # Call describe on df and transpose it due to the large number of columns
    spread = df.describe().T

    # precalculate interquartile range for ease of use in next calculation
    IQR = spread['75%'] - spread['25%']

    # create an outliers column which is either 3 IQRs below the first quartile or
    # 3 IQRs above the third quartile
    spread['outliers'] = (spread['min'] < (spread['25%'] - (3 * IQR))) | (spread['max'] > (spread['75%'] + 3 * IQR))

    # just display the features containing extreme outliers
    print(spread.loc[spread.outliers,])

    plt.rcParams['figure.figsize'] = [14, 8]
    df.meanpressure_1.hist()
    plt.title('Example: Distribution of meanpressure')
    plt.xlabel('meanpressure_1')
    plt.show()

    # iterate over the meanpressure columns
    for meanpressure_col in ['meanpressure_1', 'meanpressure_2', 'meanpressure_3']:
        # create a boolean array of values representing nans
        missing_vals = pd.isnull(df[meanpressure_col])
        df[meanpressure_col][missing_vals] = 0

    # iterate over the maxpressure columns
    for maxpressure_col in ['maxpressure_1', 'maxpressure_2', 'maxpressure_3']:
        # create a boolean array of values representing nans
        missing_vals = pd.isnull(df[maxpressure_col])
        df[maxpressure_col][missing_vals] = 0

    # iterate over the meanpressure columns
    for minpressure_col in ['minpressure_1', 'minpressure_2', 'minpressure_3']:
        # create a boolean array of values representing nans
        missing_vals = pd.isnull(df[minpressure_col])
        df[minpressure_col][missing_vals] = 0

    df = df.dropna()

    return df
    
