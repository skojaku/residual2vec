import csv
import errno
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def mk_outdir(out_path):
    """Check and make a directory.

    :param out_path: path for directory
    """
    if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    logging.info("output directory is created")
