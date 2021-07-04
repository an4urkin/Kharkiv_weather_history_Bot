import os
import pickle
import datetime
import telegram
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta
from main3 import extract_weather_data, clear_data
from linear_regression import clear_mean, clear_max, clear_min, clear_humidity
from config import API_KEY, latitude, longitude, token
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from darksky.api import DarkSky
from darksky.types import languages, units, weather

updater = Updater(token)  # Telegram API token
dispatcher = updater.dispatcher


# Telegram Bot Commands
def info_command(bot, update):
    bot.send_message(chat_id=update.message.chat.id, text='/forecastTMP - прогноз температуры на следующие 12 часов\n'
                                                          '/forecastHUM - прогноз давления на следующие 12 часов\n'
                                                          '/modelsUPD - актуализировать данные моделей\n'
                                                          '/modelsINFO - общая сводка о моделях\n'
                                                          '/info - список комманд\n')


def tomorrow_weather_command(bot, update):
    bot.send_chat_action(chat_id=update.message.chat.id, action=telegram.ChatAction.TYPING)
    date = get_current_date()
    meantemp = predict_weather_condition('meantemp', API_KEY, date, latitude, longitude)
    maxtemp = predict_weather_condition('maxtemp', API_KEY, date, latitude, longitude)
    mintemp = predict_weather_condition('mintemp', API_KEY, date, latitude, longitude)
    meanhumidity = predict_weather_condition('meanhumidity', API_KEY, date, latitude, longitude)
    meantemp = int(meantemp)
    maxtemp = int(maxtemp)
    mintemp = int(mintemp)
    meanhumidity = int(meanhumidity)
    
    nas_par = (1.0016 + 3.15 * (10 ** -6) * get_pressure() - 0.074 * (get_pressure() ** -1)) * (
                6.112 * (2.71828 ** (17.62 * meantemp / (243.12 + meantemp))))
    fakt_par = meanhumidity * nas_par / 100  # 0.451395
    preciepprob = fakt_par / nas_par * 100  # 1.00310
    bot.send_photo(chat_id=update.message.chat.id, photo=open('./jab.png', 'rb'))
    bot.send_message(chat_id=update.message.chat.id, text='🌡 Средняя температура:   +' + str(meantemp) + '°C\n'
                                                          '☀ Температура днем:      +' + str(maxtemp) + '°C\n'
                                                          '🌒 Температура ночью:      +' + str(mintemp) + '°C\n')
                                                          # '💦 Влажность:               ' + str(meanhumidity) + '%\n'
                                                          # '💦 Насыщ пар:                 ' + str(nas_par) + '%\n'
                                                          # '💦 Факт пар:                  ' + str(fakt_par) + '%\n'
                                                          # 'Вероятность осадков:      +' + str(preciepprob) + '%\n')
    pred_max = []
    darksky = DarkSky(API_KEY)
    for item in range(12):
        forecast = darksky.get_time_machine_forecast(
            latitude, longitude,
            date,
            extend=False,  # default `False`
            lang=languages.ENGLISH,  # default `ENGLISH`
            units=units.AUTO,  # default `auto`
            exclude=[weather.ALERTS]  # default `[]`
        )
        pred_max.append(forecast.hourly.data[0].temperature)
        date += timedelta(hours=1)
    i = 0
    for item in pred_max:
        bot.send_message(chat_id=update.message.chat.id, text='🌡 Температура за +'+str(i)+'-й час:   +' + str(item) + '°C\n')
        i += 1




def upload_model(type_condition):
    with open(os.path.join('./' + type_condition + '_model.pickle'), 'rb') as f:
        model = pickle.load(f)
    return model


def get_current_date():
    target_date = datetime.datetime.now() - timedelta(days=3)
    return target_date


def get_data(api_key, target_date, lat, long):
    records = extract_weather_data(api_key, target_date, lat, long, 4)
    df = clear_data(records)
    return df


def get_pressure():
    darksky = DarkSky(API_KEY)

    forecast = darksky.get_time_machine_forecast(
        latitude, longitude,
        get_current_date(),
        extend=False,  # default `False`
        lang=languages.ENGLISH,  # default `ENGLISH`
        units=units.AUTO,  # default `auto`
        exclude=[weather.ALERTS]  # default `[]`
    )
    pressure = forecast.daily.data[0].pressure
    return pressure


def predict_weather_condition(type_condition, api_key, target_date, lat, long):
    df = get_data(api_key, target_date, lat, long)
    if type_condition == 'meantemp':
        res_df = clear_mean(df)
        prediction = upload_model(type_condition).predict(res_df)
    elif type_condition == 'maxtemp':
        res_df = clear_max(df)
        prediction = upload_model(type_condition).predict(res_df)
    elif type_condition == 'mintemp':
        res_df = clear_min(df)
        prediction = upload_model(type_condition).predict(res_df)
    else:
        res_df = clear_humidity(df)
        prediction = upload_model(type_condition).predict(res_df)

    return prediction


# Handlers
info_command_handler = CommandHandler('info', info_command)
tomorrow_weather_command_handler = CommandHandler('tomorrow_weather', tomorrow_weather_command)

# Adding handlers to dispatcher
dispatcher.add_handler(info_command_handler)
dispatcher.add_handler(tomorrow_weather_command_handler)

# Looking for updates
updater.start_polling(clean=True)

