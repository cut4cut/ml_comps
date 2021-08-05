import logging


def get_logger(folder: str) -> logging.Logger:
    ff_name = './{}/learn.log'.format(folder)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(ff_name)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s] - %(asctime)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger