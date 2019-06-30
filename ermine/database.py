import os
from typing import List
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import ForeignKey, Sequence
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class Training(Base):
    __tablename__ = 'training'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    training_dir = Column(String)
    setting_file = Column(String)    
    # trial = relationship("Trial", backref="trial") 

class Trial(Base):
    __tablename__ = 'trial'
    id = Column(Integer, primary_key=True)
    seq = Column(Integer)
    trial_dir = Column(String)
    model_file = Column(String)
    training_id = Column(Integer, ForeignKey('training.id',ondelete='CASCADE'))
    status = Column(Integer) # 0 - START, 1 - TRAINING, 2 - DONE

class Sample(Base):
    __tablename__ = 'sample'
    sample_id_seq = Sequence('sample_id_seq', metadata=Base.metadata)
    id = Column(Integer, primary_key=True)
    msg = Column(String)


class DatabaseService():

    DATABASE_FILE = 'database.db'
    home = None

    def __init__(self, home:str):
        self.home = home
        self.database_file = self.home + os.path.sep + DatabaseService.DATABASE_FILE

    def delete_database_file(self):
        os.remove(self.database_file)

    def prepare_database(self):
        engine = create_engine("sqlite:///" + self.database_file)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
    
    def find_training(self, name:str)->List[Training]:
        print('find training')
        training = self.session.query(Training).filter(Training.name==name).all()
        print('training', training)
        return training

    def create_new_training(self,name:str)->Training:
        # 設定ファイル名は、home_dir/training/name/name.json
        length = len(self.find_training(name))
        if length != 0:
            raise Exception("Training already exists.")
        
        training_dir = self.home + os.path.sep + 'training' + os.path.sep + name
        setting_file_path = training_dir + os.path.sep + name + '_setting.json'
        print(training_dir)
        self.session.add(Training(name=name, training_dir=training_dir, setting_file=setting_file_path))
        self.session.commit()
        training = self.session.query(Training).filter(Training.name==name)[0]
        return training
    
    def find_trials(self, parent:Training)->List[Trial]:
        trials = self.session.query(Trial).filter(Trial.id==parent.id).all()
        return trials

    def create_new_trial(self, parent:Training):
        trials = self.find_trials(parent)
        seq = len(trials)
        trial_dir = parent.training_dir + os.path.sep + str(seq)
        model_file = trial_dir + os.path.sep + 'model.hdf5'
        traial_setting = trial_dir + os.path.sep + 'trial_parameters.json'
        self.session.add(Trial(seq=seq,trial_dir=trial_dir,model_file=model_file,training_id=parent.id,status=0))
        trial= self.session.query(Trial).filter(Trial.training_id==parent.id,Trial.seq==seq)[0]
        return trial

    def add_sample(self):
        self.session.add(Sample(msg='Hello'))
        self.session.add(Sample(msg='GoodBy'))
        self.session.commit()

        samples = self.session.query(Sample)
        for sample in samples:
            print('Found ', sample.msg)

        self.session.add(Training(name='aaa',setting_file='hoge.json'))
        self.session.commit()
        trainings = self.session.query(Training).filter(Training.name=='aaa')
        for training in trainings:
            print('Found ', training.name)

   
