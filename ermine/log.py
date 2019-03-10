from enum import Enum
import logging


class LogLevel(Enum):
    DEBUG: int = logging.DEBUG
    INFO: int = logging.INFO
    WARNING: int = logging.WARNING
    ERROR: int = logging.ERROR


class LogUtil():
    @staticmethod
    def init(name: str = __name__, level: int = LogLevel.DEBUG):
        LogUtil.logger = logging.getLogger(name)
        LogUtil.logger.setLevel(level)

    @staticmethod
    def debug(s: str):
        LogUtil.logger.debug(s)

    @staticmethod
    def info(s: str):
        LogUtil.logger.info(s)

    @staticmethod
    def warn(s: str):
        LogUtil.logger.warn(s)

    @staticmethod
    def error(s: str):
        LogUtil.logger.error(s)
