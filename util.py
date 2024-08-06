import os
from math import sqrt, floor, ceil
import numpy as np
import base64
import cv2
import re
import logging
from logging.handlers import TimedRotatingFileHandler
# from discordwebhook import Discord
from pyaml_env import parse_config

CONFIG = parse_config('config/config.yaml')
FORMATTER = logging.Formatter("%(asctime)s -- %(levelname)s -- %(message)s")
BACKUP_COUNT = 672

def get_midpoint(rect):
    x1, y1, x2, y2 = rect
    mid_x = (x2 + x1) / 2
    mid_y = (y2 + y1) / 2
    return (int(mid_x), int(mid_y))


def get_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    d1 = pow(x2-x1, 2)
    d2 = pow(y2-y1, 2)
    distance = sqrt(d1 + d2)
    return distance


def decode(str_buffer):
    byte_buffer = str_buffer.encode()
    byte_buffer = base64.b64decode(byte_buffer)
    buffer = np.frombuffer(byte_buffer, dtype=np.uint8)
    img_arr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return img_arr


def crop_image(image, window):
    window1 = floor(window/2)
    window2 = ceil(window/2)
    w = image.shape[1]
    x1 = y1 = int((w/2) - window1)
    x2 = y2 = int((w/2) + window2)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

class CustomLogger(logging.Logger):
    def critical(self, msg, *args, **kwargs):
        # add custom behavior
        # send_notification()
        # discord.post(content=msg)
        super().critical(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        # discord.post(content=msg)
        super().warning(msg, *args, **kwargs)

class LOGGER(object):
    def __init__(self, name, loglevel=logging.DEBUG, log_path=None):
        self.logger = CustomLogger(name)
        self.logger.setLevel(loglevel)
        if log_path:
            self.file_handler = TimedRotatingFileHandler(os.path.join(log_path, name), when="M", interval=15, backupCount=BACKUP_COUNT)
        else:
            self.file_handler = TimedRotatingFileHandler(f"{name}.log", when="M", interval=15, backupCount=BACKUP_COUNT)
        self.file_handler.setFormatter(FORMATTER)
        self.logger.addHandler(self.file_handler)


DIR_PATH = CONFIG["paths"]["root"]
BACKUP_COUNT = 672


class WrongImageFormat(Exception):
    """Exception raised for errors in the image format.
    Attributes:
        item -- input part and message where the error occured
    """
    def __init__(self, item):
        self.part = next(iter(item))
        self.message = next(iter(item.values()))
        super().__init__(self.part, self.message)


def contains_letters_in_order(word, letters):
    regex = '.*'.join(map(re.escape, letters))
    return re.search(regex, word) is not None


def contains_specific_numbers(word, numbers):
    check = False
    if str(word[0]) == numbers[0]:
        if word[3:6] == numbers[1:]:
            check = True
    return check


def save_error_extention(date, destination, part, lot):
    error_path = os.path.join(DIR_PATH, destination,
                              date, part, lot, "save_error_extention.txt")
    f = open(error_path, "w+")
    f.close()


def encode(img_arr):
    _, buffer = cv2.imencode(".jpeg", img_arr)
    byte_buffer = base64.b64encode(buffer)
    str_buffer = byte_buffer.decode()
    return str_buffer


def color_merge(blue, green, red):
    b = cv2.imread(blue, 0)
    g = cv2.imread(green, 0)
    r = cv2.imread(red, 0)
    img = cv2.merge((b, g, r))
    return img