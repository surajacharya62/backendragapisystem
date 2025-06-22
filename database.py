

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

url_database = "mysql+pymysql://user_developer:Developer#321@127.0.0.1:3306/ragdb"

engine = create_engine(url_database)

Sessionlocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

Base = declarative_base()
