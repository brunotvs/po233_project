import csv
import datetime

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql import functions

from models import models, Base

engine = create_engine('sqlite:///sql/data.db')
Session = sessionmaker(bind=engine)
session = Session()
Base.metadata.create_all(engine)

reservoir_data = {}
with open(r'data\raw\reservatorio_vazao.csv', 'r') as file:
    reader = csv.reader(file)

    next(reader, None)  # Skip header
    for row in reader:
        date = datetime.date.fromisoformat(row[0])
        reservoir_data[date] = row[2]


with open(r'data\raw\reservatorio_nivel.csv', 'r') as file:
    reader = csv.reader(file)

    next(reader, None)  # Skip header
    for row in reader:
        date_string: str = row[0].strip()
        m, d, y = [int(x)for x in date_string.split('/')]
        date = datetime.date(y, m, d)
        reservoir_data[date] = [reservoir_data[date], row[2]]


for key, value in reservoir_data.items():
    reservoir = models.Reservoir(key, level=value[1], streamflow=value[0])
    session.add(reservoir)

session.commit()
