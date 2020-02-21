import logging
logging.basicConfig(format="[%(asctime)s %(name)s %(filename)s %(funcName)s %(lineno)d %(levelname)s] %(message)s",
                    level=logging.INFO)
from logging.handlers import RotatingFileHandler

def init_logger(log_name, log_file=None, log_level=logging.INFO, log_file_level=logging.INFO):

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(log_format)
    # logger.addHandler(console_handler)

    if log_file and log_file != '':
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1000, backupCount=10)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)

    return logger