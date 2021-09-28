from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship, sessionmaker

engine = create_engine("postgresql://user:password@192.168.2.244:5432/my_database")
Session = sessionmaker(bind=engine)
Base = declarative_base()
