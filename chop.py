import sys
from glob import glob
import cv2
import numpy as np
import statistics as stat
import math
import os

# from predict import get_model_api
# attention_ocr = get_model_api()

# image_line = cv2.imread('image.jpg',0)
# out_text = attention_ocr(image_line)


def illumination_correction(image):
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(25,15))
    image = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,_structure)
    image = cv2.bitwise_not(image)
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,_structure)
    return image


def threshold(image):
    image = cv2.threshold(image,128,255,cv2.THRESH_OTSU)[1]
    return image

def get_lines_segments(line_image):
    line_seg = []
    _line_image = line_image.copy()
    _line_image = illumination_correction(_line_image)
    cv2.imwrite('tmp/illumin.jpg', _line_image)
    _line_image = cv2.cvtColor(_line_image,cv2.COLOR_BGR2GRAY)
    _line_image = threshold(_line_image)
    cv2.imwrite('tmp/thresh.jpg', _line_image)
    _width_limit = 250
    hist = cv2.reduce(_line_image,0, cv2.REDUCE_AVG).reshape(-1)
    max_th = max(hist) * .8
    H, W = _line_image.shape[:2]
    th = max_th
    to = [y for y in range(W) if hist[y] >= th]
    _from = 0
    _to = 1

    for i,x in enumerate(to,1):
        # cv2.line(line, (x,0), (x,H),(0,0,255), 1)
        _width = x - _from
        if  _width <= _width_limit: 
            _to = x

        else:
            _line = line_image[:,_from:_to]
            _from = _to
            _to = x
            if _line.shape[1] > 0 and _line.shape[0] > 0:
                _line = cv2.copyMakeBorder(_line,0,0,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
                line_seg.append(_line)
            
        if x == to[-1] and x != _from:
            _line = line_image[:,_from:]
            if _line.shape[1] > 0 and _line.shape[0] > 0:
                _line = cv2.copyMakeBorder(_line,0,0,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
                line_seg.append(_line)

    return line_seg

def chop(img_for_ocr):
    i = 0
    h, w = img_for_ocr.shape[:2]
    print('image width', w)
    if w > 350:
        line_segments = get_lines_segments(img_for_ocr)
        for j, _line in enumerate(line_segments):
            out_txt = 
            cv2.imwrite('tmp/{}_{}.jpg'.format(i, j), _line)


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    chop(image)