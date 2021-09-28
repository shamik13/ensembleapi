from datetime import date

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    String,
    Table,
    create_engine,
)
from sqlalchemy.orm import session

from base import Base, Session

engine = create_engine("postgresql://user:password@192.168.2.244:5432/db")
session = Session()


class InfoTable(Base):
    __tablename__ = "infotable"
    idx = Column(String, primary_key=True)
    dataset_name = Column(String)
    model_name = Column(String)
    roc_score = Column(String)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
