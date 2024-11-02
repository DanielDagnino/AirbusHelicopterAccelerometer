import cv2
import numpy as np
from PIL import Image


def rmv_black_border(img):
    # img = np.uint8(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bb = [cv2.boundingRect(_) for _ in contours]
    area = [_[2] * _[3] for _ in bb]
    if not area:
        img = img[:2, :2]
    else:
        idx_max_area = np.argmax(area).item()
        x, y, w, h = cv2.boundingRect(contours[idx_max_area])
        img = img[y:y + h, x:x + w]

    # img = Image.fromarray(img)
    return img, x, y, w, h


def resize_crop_save(image_path, output_path, size_hw=(100, 100)):
    img = Image.open(image_path)

    width, height = img.size
    if width < height:
        new_height = int((size_hw[1] / width) * height)
        img = img.resize((size_hw[0], new_height), Image.LANCZOS)
    else:
        new_width = int((size_hw[0] / height) * width)
        img = img.resize((new_width, size_hw[1]), Image.LANCZOS)

    left = (img.width - size_hw[1]) / 2
    top = (img.height - size_hw[0]) / 2
    right = (img.width + size_hw[1]) / 2
    bottom = (img.height + size_hw[0]) / 2
    left, top, right, bottom = [int(_) for _ in [left, top, right, bottom]]
    img = img.crop((left, top, right, bottom))

    img.save(output_path)
