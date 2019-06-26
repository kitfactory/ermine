# import pytest
import os
import shutil
import sys
import tensorflow as tf
# from os.path import expanduser


from .. base import HomeFile
from .. database import DatabaseService

def test_database():
    print(sys.path)

    home = HomeFile.get_home_dir()
    database = DatabaseService(home)
    database.prepare_database()
    database.add_sample()
    assert(True)
