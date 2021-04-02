from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Integer,
                        Sequence, String, Table, Time)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Index, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import false, null, true

from . import Base, session


class Coordinate(Base):
    __tablename__ = 'coordinates'

    id = Column(Integer, nullable=False, unique=True, primary_key=True, autoincrement=True)
    latitude = Column(Float, nullable=False, unique=False)
    longitude = Column(Float, nullable=False, unique=False)

    river_id = Column(Integer, ForeignKey("rivers.id"), nullable=False)
    river = relationship('River', back_populates='coordinates', uselist=False)

    variables = relationship('Variables', back_populates='coordinate')

    UniqueConstraint('latitude', 'longitude', name='unique_coordinate')

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = round(latitude, 5)
        self.longitude = round(longitude, 5)
        session.add(self)


class River(Base):
    __tablename__ = 'rivers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True)
    coordinates = relationship('Coordinate', back_populates='river')

    def __init__(self, name: str) -> None:
        self.name = name
        session.add(self)


class Variables(Base):
    __tablename__ = 'variables'

    id = Column(Integer, primary_key=True, unique=True, autoincrement=True)
    date = Column(Date)
    time = Column(Time)
    value = Column(Float)
    variable = Column(String, nullable=False)
    scenario = Column(String, nullable=False)

    coordinate_id = Column(String, ForeignKey('coordinates.id'), nullable=False)
    coordinate = relationship('Coordinate', back_populates='variables', uselist=False)

    def __init__(self, value, date, time, variable: str, scenario: str) -> None:
        self.value = value
        self.date = date
        self.time = time
        self.variable = variable
        self.scenario = scenario
        session.add(self)
