from ermine import ErmineLogger

import unittest
import logging

import time

class ErmineLoggerTestCase(unittest.TestCase):

    def test_log(self):
        logger = ErmineLogger.get_instance()
        logger.debug("DEBUG MESSAGE")
        logger.info("INFO MESSAGE")
        logger.warn(" MESSAGE")
        logger.error("DEBUG MESSAGE")

        self.assertTrue(True)


if __name__ == '__main__':
    logger = ErmineLogger.get_instance()
    logger.debug("DEBUG MESSAGE")

