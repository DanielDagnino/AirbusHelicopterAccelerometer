import inspect
import logging

import cv2
from PIL import Image


def img_read_PIL_RGB(fn: str) -> Image.Image:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
    try:
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise IOError(f"Error reading image using cv2 in color scale: {fn}")
        img = Image.fromarray(img)
    except Exception as expt1:
        try:
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[0] == 0 or img.shape[1] == 0:
                raise IOError(f"Error reading image using cv2 in gray scale: {fn}")
            img = Image.fromarray(img)
        except Exception as expt2:
            try:
                img = Image.open(fn)
                img = img.convert('RGB')
            except Exception as expt3:
                logger.error(f'An error occurred while loading image file with PIL: {fn}')
                logger.error(expt1)
                logger.error(expt2)
                logger.error(expt3)
                raise RuntimeError(f"Error reading image using CV and PIL: {fn}")
    return img
