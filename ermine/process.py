from subprocess import Popen
from shlex import split

class ProcessUtil():

    def __init__(self):
        super().__init__()
        self.train_process = None
        self.evaluate_process = None
        self.tensorboard_process = None

    def execute_training(self):
        self.train_process = Popen(split(cmd))

    def is_training(self)->bool:
        if( self.train_process == None):
            return False
        if( self.train_process.poll()== None):
            return True
        else:
            return False
    
    def kill_training(self):
        if( self.train_process == None):
            return
        self.train_process.kill()
        self.train_process.wait()
        self.train_process = None

    def execute_evaluation(self):
        self.train_process = Popen(split(cmd))

    def is_evaluating(self)->bool:
        if( self.train_process == None):
            return False
        if( self.train_process.poll()== None):
            return True
        else:
            return False

    def kill_evaluation(self):
        if( self.evaluate_process == None):
            return
        self.evaluate_process.kill()
        self.evaluate_process.wait()
        self.evaluate_process = None
    
    def execute_tensorboard(self):
        if( self.tensorboard_process != None):
            return
        self.tensorboard_process = Popen(split("tensorboard --logdir=."))
        print('execute done?')
    
    def is_tensorboard_running(self):
        if( self.tensorboard_process == None):
            return False
        if( self.tensorboard_process.poll()== None):
            return True
        else:
            return False

    def kill_tensorboard(self):
        if( self.tensorboard_process == None):
            return
        self.tensorboard_process.kill()
        self.tensorboard_process.wait()
        self.tensorboard_process = None

    def kill_all(self):
        self.kill_training()
        self.kill_evaluation()
        self.kill_tensorboard()