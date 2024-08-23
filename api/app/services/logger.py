import logging
import sys


class Logger:

    def __init__(self):
        self._setLogger()

    def _setLogger(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s: [%(levelname)s] %(message)s'
        )

        handler.setFormatter(formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)

        return self.logger
