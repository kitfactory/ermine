import logging
import inspect

import os
from os.path import expanduser


try:
    import tensorflow
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.getLogger().removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass


class ErmineLogger():
    DEBUG: int = logging.DEBUG
    INFO: int = logging.INFO
    WARNING: int = logging.WARNING
    ERROR: int = logging.ERROR
    instance:'ErmineLogger' = None

    def __init__(self):
        self.logger:logging.Logger = logging.getLogger('Ermine')
        home = expanduser("~")

        if os.path.exists(home + os.path.sep + "ermine_debug_mode"):
            self.logger.setLevel(ErmineLogger.DEBUG)
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)
        else:
            self.logger.setLevel(ErmineLogger.WARNING)
            handler = logging.FileHandler(home+ os.path.sep+ "ermine_log.txt")
            self.logger.addHandler(handler)
    
    def debug(self, msg: str):
        """ log debug message
        
        Args:
            msg (str): information debug message

        """
        frame = inspect.currentframe().f_back
        f = frame.f_code.co_filename
        n = frame.f_lineno
        s = "[DEBUG] {file} {no} : {msg}".format(file=f,no=n,msg=msg)
        self.logger.debug(s)

    def info(self, msg: str):
        """ log information message
        
        Args:
            msg (str): information level message
        """        
        frame = inspect.currentframe().f_back
        f = frame.f_code.co_filename
        n = frame.f_lineno
        s = "[INFO] {file} {no} : {msg}".format(file=f,no=n,msg=msg)
        self.logger.info(s)

    def warn(self, msg: str):
        """ log wraning message
        
        Args:
            msg (str): warning level message
        """        
        frame = inspect.currentframe().f_back
        f = frame.f_code.co_filename
        n = frame.f_lineno
        s = "[WARN] {file} {no} : {msg}".format(file=f,no=n,msg=msg)
        self.logger.warn(s)

    def error(self, msg: str):
        """log error message
        
        Args:
            msg (str): error level message
        """        
        frame = inspect.currentframe().f_back
        f = frame.f_code.co_filename
        n = frame.f_lineno
        s = "[ERROR] {file} {no} : {msg}".format(file=f,no=n,msg=msg)
        self.logger.error(s)


    def excepiton(self, exc:Exception):
        self.logger.exception(exc)

    @staticmethod
    def get_instance()->'ErmineLogger':
        """get instance of the ermine logger
        
        Returns:
            ErmineLogger: logger object
        """
        if ErmineLogger.instance is None:
            ErmineLogger.instance = ErmineLogger()
        return ErmineLogger.instance