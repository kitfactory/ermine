import os
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Sequence
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

'''
class Training(Base):
    __tablename__ = 'training'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    setting_file = Column(String)
    trial = relationship("Trial", backref="trial") 

class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)
    seq = Column(Integer)
    model_file = Column(String)
    train_id = Column(Integer, ForeignKey('training.id'))
    status = Column(Integer) # 0 - START, 1 - TRAINING, 2 - DONE
    user = relationship("User") 
'''


class Sample(Base):
    __tablename__ = 'sample'

    sample_id_seq = Sequence('sample_id_seq', metadata=Base.metadata)
    id = Column(Integer, primary_key=True)
    msg = Column(String)


class DatabaseService():

    DATABASE_FILE = 'database.db'

    def __init__(self, home:str):
        self.database_file = home + os.path.sep + DatabaseService.DATABASE_FILE

    def prepare_database(self):
        engine = create_engine("sqlite:///" + self.database_file)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
    
    def add_sample(self):
        self.session.add(Sample(msg='Hello'))
        self.session.add(Sample(msg='GoodBy'))
        self.session.commit()

        samples = self.session.query(Sample)
        for sample in samples:
            print('Found ', sample.msg)
