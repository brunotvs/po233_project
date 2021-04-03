import datetime

import shapefile
from requests import models

from source.build_data.data_downloader import projeta
from source.build_data.models import models, session

import concurrent.futures

# lista de identificadores de cenários como apresentado no projeta
# passar o número correspondente para a função
# scenarios = {
#     1: "20 km, RCP4.5, continental, MIROC5.",
#     2: "20 km, RCP8.5, continental, MIROC5.",
#     3: "20 km, RCP4.5, continental, HADGEM2-ES.",
#     4: "05 km, RCP4.5, sudeste-BR, HADGEM2-ES.",
#     5: "20 km, RCP8.5, continental, HADGEM2-ES.",
#     6: "05 km, RCP8.5, sudeste-BR, HADGEM2-ES.",
#     7: "20 km, RCP4.5, continental, CANESM2.",
#     8: "20 km, RCP8.5, continental, CANESM2.",
#     13: "20 km, Histórico, continental, CANESM2.",
#     14: "20 km, Histórico, continental, HADGEM2-ES.",
#     15: "20 km, Histórico, continental, MIROC5.",
#     16: "05 km, Histórico, sudeste-BR, HADGEM2-ES.",
#     17: "20 km, Histórico, continental, BESM.",
#     19: "20 km, RCP8.5, continental, BESM.",
#     20: "20 km, RCP4.5, continental, BESM.",
#     21: "05 km, Histórico, sudesteD2-BR, HADGEM2-ES.",
#     22: "05 km, RCP4.5, sudesteD2-BR, HADGEM2-ES.",
#     23: "05 km, RCP8.5, sudesteD2-BR, HADGEM2-ES.",
# }

# Lista de frequências, passar o número correspondente
# frequencies = {
#     4: "YEARLY",
#     3: "MONTHLY",
#     2: "DAILY",
#     1: "HOURLY"
# }

shapefile_path = 'data/shapefile/buffer_coord'


rcp4_5 = [
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 4,
     'frequency': 2,
     'variable': 'PREC'},
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 4,
     'frequency': 2,
     'variable': 'TP2M'},
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 4,
     'frequency': 2,
     'variable': 'EVAP'},
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 4,
     'frequency': 2,
     'variable': 'RNOF'}
]

rcp8_5 = [
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 6,
     'frequency': 2,
     'variable': 'PREC'},
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 6,
     'frequency': 2,
     'variable': 'TP2M'},
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 6,
     'frequency': 2,
     'variable': 'EVAP'},
    {'start_month': 1,
     'star_year': 2006,
     'end_month': 12,
     'end_year': 2100,
     'scenario': 6,
     'frequency': 2,
     'variable': 'RNOF'}
]

historic = [
    {'start_month': 1,
     'star_year': 1999,
     'end_month': 12,
     'end_year': 2005,
     'scenario': 16,
     'frequency': 2,
     'variable': 'PREC'},
    {'start_month': 1,
     'star_year': 1999,
     'end_month': 12,
     'end_year': 2005,
     'scenario': 16,
     'frequency': 2,
     'variable': 'TP2M'},
    {'start_month': 1,
     'star_year': 1999,
     'end_month': 12,
     'end_year': 2005,
     'scenario': 16,
     'frequency': 2,
     'variable': 'EVAP'},
    {'start_month': 1,
     'star_year': 1999,
     'end_month': 12,
     'end_year': 2005,
     'scenario': 16,
     'frequency': 2,
     'variable': 'RNOF'}
]

downloads = rcp4_5 + rcp8_5 + historic


def main():

    parse_shapefile()

    # print(get_projeta_data_async(**downloads[0], latitude=-13, longitude=-49))
    parse_projeta_data_from_downloads_list()

    print('4')


def parse_shapefile():
    records = shapefile.Reader(shapefile_path).records()
    coordinates, rivers = [[record[0], record[1]] for record in records], set([record[2] for record in records])

    sorted_rivers = sorted(rivers)
    for river in sorted_rivers:
        models.River(river)

    for i, coordinate in enumerate(coordinates):
        river = session.query(models.River).filter(models.River.name == records[i][2]).one()
        river.coordinates.append(models.Coordinate(coordinate[0], coordinate[1]))

    session.commit()


def parse_projeta_data_from_downloads_list():
    coordinates = session.query(models.Coordinate).all()[:1]

    for download in downloads:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(build_variables_table, coordinates, [download for _ in coordinates])

            for projeta_data, coordinate in results:
                for data in projeta_data:
                    print(data)
                    try:
                        date = parse_projeta_date(data['date'])
                    except ValueError:
                        continue

                    try:
                        time = parse_projeta_time(data['time'])
                    except ValueError:
                        continue
                    coordinate.variables.append(
                        models.Variables(
                            value=data['value'],
                            date=date,
                            time=time,
                            scenario=download['scenario'],
                            variable=download['variable']))

    session.commit()

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     projeta_data = [executor.submit(
    #         build_variables_table, {
    #             'coordinate': coordinate,
    #             **download}) for coordinate in coordinates]

    #     for _ in projeta_data:
    #         session.commit()


def get_projeta_data_async(
        start_month,
        star_year,
        end_month,
        end_year,
        scenario,
        frequency,
        variable,
        latitude,
        longitude):

    sliced_periods = projeta.slice_period(
        start_month=start_month,
        start_year=star_year,
        end_month=end_month,
        end_year=end_year)

    urls = projeta.build_urls(
        scenario=scenario,
        frequency=frequency,
        variable=variable,
        latitude=latitude,
        longitude=longitude,
        periods=sliced_periods)
    return projeta.get_data_async(urls)


def parse_projeta_date(date: str) -> datetime.date:
    day, month, year = date.split('/')
    return datetime.date(int(year), int(month), int(day))


def parse_projeta_time(time: str) -> datetime.time:
    return datetime.time.fromisoformat(time)


def build_variables_table(coordinate: models.Coordinate, download):
    projeta_data = get_projeta_data_async(
        **download,
        latitude=coordinate.latitude,
        longitude=coordinate.longitude)

    return projeta_data, coordinate


if __name__ == '__main__':
    main()
