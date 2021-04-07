import concurrent.futures
import datetime
import time

import shapefile
from requests import models
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.sql import functions
from sqlalchemy.orm.exc import NoResultFound

from source.build_data.data_downloader import projeta
from source.build_data.models import engine, models, session, create_db, remove_db

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

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

rcp_common = {
    'start_month': 1, 'star_year': 2006,
    'end_month': 12, 'end_year': 2099,
    'frequency': 2,
}

historic_common = {
    'start_month': 1, 'star_year': 1999,
    'end_month': 12, 'end_year': 2005,
    'frequency': 2,
}

rcp4_5 = [
    {'scenario': 4, 'variable': 'PREC'},
    # {'scenario': 4, 'variable': 'TP2M'},
    # {'scenario': 4, 'variable': 'EVTP'},
    # {'scenario': 4, 'variable': 'RNOF'}
]

[rcp4_5_d.update(rcp_common) for rcp4_5_d in rcp4_5]

rcp8_5 = [
    {'scenario': 6, 'variable': 'PREC'},
    {'scenario': 6, 'variable': 'TP2M'},
    {'scenario': 6, 'variable': 'EVTP'},
    {'scenario': 6, 'variable': 'RNOF'}
]

[rcp8_5_d.update(rcp_common) for rcp8_5_d in rcp8_5]

historic = [
    {'scenario': 16, 'variable': 'PREC'},  # done
    # {'scenario': 16, 'variable': 'TP2M'},  # done
    # {'scenario': 16, 'variable': 'EVTP'},  # done
    # {'scenario': 16, 'variable': 'RNOF'}  # done
]

[historic_d.update(historic_common) for historic_d in historic]

downloads = []
# downloads += rcp4_5
# downloads += rcp8_5
downloads += historic


def main():
    remove_db()
    create_db()

    parse_shapefile()
    # print(get_projeta_data_async(**downloads[0], latitude=-13, longitude=-49))
    parse_projeta_data_from_downloads_list()

    Session.remove()


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

    variables = {
        'PREC': 'precipitation',
        'TP2M': 'temperature',
        'EVTP': 'evaporation',
        'RNOF': 'surface_runoff',
    }
    for download in downloads:
        with concurrent.futures.ThreadPoolExecutor(50) as executor:
            results = executor.map(build_variables_table, coordinates, [download for _ in coordinates])
            coordinates_count = session.query(functions.count(models.Coordinate.id)).scalar()
            print(coordinates_count)
            print_progress_bar(0, coordinates_count, f"{download['scenario']} - {download['variable']}")

            for i, (projeta_data, coordinate) in enumerate(results):
                th_session = Session()
                th_coordinate = th_session.query(models.Coordinate).get(coordinate.id)
                print_progress_bar(
                    i + 1,
                    coordinates_count,
                    f"{download['scenario']} - {download['variable']}",
                    f'{i+1} {coordinate.latitude} {coordinate.longitude}')

                for data in projeta_data:
                    try:
                        date = parse_projeta_date(data['date'])
                    except ValueError:
                        continue

                    try:
                        time = parse_projeta_time(data['time'])
                    except ValueError:
                        continue

                    try:
                        var = th_session.query(models.Variables)\
                            .filter(models.Variables.coordinate == coordinate)\
                            .filter(models.Variables.date == date)\
                            .filter(models.Variables.time == time)\
                            .filter(models.Variables.scenario == download['scenario'])\
                            .one()
                        print('var found')
                    except NoResultFound:
                        var = models.Variables(
                            date=date,
                            time=time,
                            scenario=download['scenario'],
                            coordinate=th_coordinate)
                        print('var not found')

                    setattr(var, variables[download['variable']], data['value'])

                    th_session.add(var)

                th_session.commit()

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
    if '/' in date:
        day, month, year = date.split('/')
    else:
        year, month, day = date.split('-')
    return datetime.date(int(year), int(month), int(day))


def parse_projeta_time(time: str) -> datetime.time:
    return datetime.time.fromisoformat(time)


def build_variables_table(coordinate: models.Coordinate, download):
    projeta_data = get_projeta_data_async(
        **download,
        latitude=coordinate.latitude,
        longitude=coordinate.longitude)

    return projeta_data, coordinate


def print_progress_bar(
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=1,
        length=100,
        fill='█',
        printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print('\n', time.perf_counter() - start)
