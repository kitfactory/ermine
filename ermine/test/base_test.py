# import pytest
import os
import shutil
import sys
# from os.path import expanduser


from .. base import HomeFile, Sequence, ErmineRunner, Bucket


def test_home_dir():
    print(sys.path)

    home = HomeFile.get_home_dir()
    if os.path.exists(home):
        shutil.rmtree(home)
    HomeFile.make_home_dir()
    assert os.path.exists(home)


def test_sequencer():
    home = HomeFile.get_home_dir()
    path = home + os.path.sep + Sequence.SEQUENCE_FILE
    if os.path.exists(path):
        os.remove(path)
    seq = Sequence.get_sequcene()
    assert seq == 0

    seq = Sequence.get_sequcene()
    assert seq == 1

    seq = Sequence.get_sequcene()
    assert seq == 2


def test_run():
    runner: ErmineRunner = ErmineRunner()
    runner.set_config(json_file='ermine/test/test_run.json')
    bucket: Bucket = runner.run()
    assert bucket['Message'] == 'HelloWorld'


def test_overwrite():
    runner: ErmineRunner = ErmineRunner()
    runner.set_config(json_file='ermine/test/test_run.json')
    new_opt = {'Message': 'OverwriteKey'}
    runner.overwrite_units_block_config(new_opt)
    bucket: Bucket = runner.run()
    assert bucket['OverwriteKey'] == 'HelloWorld'


def test_global():
    runner: ErmineRunner = ErmineRunner()
    runner.set_config(json_file='ermine/test/test_global.json')
    bucket: Bucket = runner.run()
    assert bucket['Message'] == 'HelloWorld'

# def test_classification():
#    runner: ErmineRunner = ErmineRunner()
#    runner.set_config(json_file='ermine/test/test_classification.json')
#    bucket: Bucket = runner.run()
#    assert bucket['Message'] == 'HelloWorld'
