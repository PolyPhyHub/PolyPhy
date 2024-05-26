# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

from datetime import datetime
import logging
import time
import traceback
from logging.handlers import RotatingFileHandler


class Logger:
    @staticmethod
    def stamp() -> str:
        return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    log_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    # STDOUT:
    console_logger = logging.getLogger("console")
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_logger.addHandler(console_handler)
    console_log_functions = {
        'debug': console_logger.debug,
        'info': console_logger.info,
        'warning': console_logger.warning,
        'error': console_logger.error,
        'critical': console_logger.critical
    }

    @staticmethod
    def logToStdOut(level, *msg) -> None:
        Logger.console_logger.setLevel(Logger.log_level.get(level, logging.INFO))
        res = " ".join(map(str, msg))
        Logger.previous_log_message = res
        Logger.console_log_functions.get(level, Logger.console_logger.info)(res)

    # FILE:
    file_logger = logging.getLogger("file")
    file_handler = logging.FileHandler("polyphy.log")
    file_handler = RotatingFileHandler("polyphy.log", maxBytes=1024*1024, backupCount=3)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    file_log_functions = {
        'debug': file_logger.debug,
        'info': file_logger.info,
        'warning': file_logger.warning,
        'error': file_logger.error,
        'critical': file_logger.critical
    }

    @staticmethod
    def logToFile(level, *msg) -> None:
        Logger.file_logger.setLevel(Logger.log_level.get(level, logging.INFO))
        res = " ".join(map(str, msg))
        Logger.file_log_functions.get(level, Logger.file_logger.info)(res)

    @staticmethod
    def logException(level, exception, *msg) -> None:
        log_msg = " ".join(map(str, msg))
        log_msg += f"\nException: {repr(exception)}"
        log_msg += f"\nStack Trace: {traceback.format_exc()}"
        
        # Log to console
        Logger.logToStdOut(level, log_msg)
        
        # Log to file
        Logger.logToFile(level, log_msg)
