import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


def create_db():
    from .models import Coordinate, Variables, River
    Base.metadata.create_all(engine)


def remove_db():
    try:
        os.remove("data.db")

    except BaseException:
        pass


create_db()
