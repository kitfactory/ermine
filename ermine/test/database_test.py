# import pytest
import os
import shutil
import sys
import tensorflow as tf
# from os.path import expanduser


from .. base import HomeFile
from .. database import DatabaseService

def test_database():
    home = HomeFile.get_home_dir()
    database = DatabaseService(home)
    database.delete_database_file()
    database.prepare_database()
    training = database.find_training('Hoge')
    assert(len(training)==0)
    training = database.create_new_training('NewTraining')
    print('NewTraining id ', training.id)
    assert(training.name=='NewTraining')

    try:
        training2 = database.create_new_training('NewTraining')
        print('add training2!!')
        assert(training2!= None)
    except Exception as exp:
        print('add failed training2!!')
        assert(True)
    
    trial = database.find_trials(training)
    assert(len(trial)==0)
    trial = database.create_new_trial(training)
    assert(trial.seq==0)

    # training = database.create_new_training('Test')
