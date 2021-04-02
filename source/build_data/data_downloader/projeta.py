import typing
from typing import Tuple
import requests
import concurrent.futures
import datetime

from sqlalchemy.sql.sqltypes import Integer

projeta_url = 'https://projeta.cptec.inpe.br/api/v1/public'


def get_frequency(frequency):
    frequencies = {
        4: "YEARLY",
        3: "MONTHLY",
        2: "DAILY",
        1: "HOURLY"
    }

    return frequencies[frequency]


def get_data(url):
    print(url)
    response = requests.get(url).json()
    # print(response.text)

    data = []
    for v in response:
        data.append({'value': v['value'], 'date': v['date'], 'time': v['time']})
    return data


def build_urls(scenario, frequency, variable, latitude, longitude, periods):
    url_scenario = f'ETA/{scenario}'
    url_frequency = f'{get_frequency(frequency)}/{frequency}'
    url_variable = f'{variable}'
    url_coordinate = f'{latitude}/{longitude}'

    urls = []
    for start_month, start_year, end_month, end_year in periods:
        full_url = f'{projeta_url}/{url_scenario}/{url_frequency}/{start_month}/{start_year}/{end_month}/{end_year}/{url_variable}/{url_coordinate}'
        urls.append(full_url)
    return urls


def slice_period(start_month, start_year, end_month, end_year):
    periods = []
    increment = 0
    if start_year == end_year:
        return [(start_month, start_year, end_month, end_year)]
    for i, year in enumerate(range(start_year, end_year - 1)):
        increment = 0 if i == 0 else 1
        correct_start_month = start_month if i == 0 else (start_month + increment) % 12
        periods.append((correct_start_month, year + (start_month + 1) // 12, start_month, year + 1))

    periods.append(((start_month + increment) % 12, end_year - 1 + (start_month + 1) // 12, end_month, end_year))

    return periods


def get_data_async(urls):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        datasets = executor.map(get_data, urls)

        data = []
        for d in datasets:
            data += d

    return data
