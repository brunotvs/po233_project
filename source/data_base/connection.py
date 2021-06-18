from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///sql/data.db')
Session = sessionmaker(bind=engine)
session = Session()
