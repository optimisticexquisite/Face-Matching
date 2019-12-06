import re

import cv2
import pytesseract
from src.OCR.regex_patterns import IC_PATTERNS, DRIVING_PATTERN, IC_NUMBER_REGREX, \
    DRIVING_DATE_REGREX, DRIVING_IC_NUMBER_REGREX, PASSPORT_DATE_REGREX, PASSPORT_PATTERNS

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'D:/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

IDENTITY_CARD = "IDENTITY CARD"
DRIVING_LICENSE = "DRIVING LICENSE"
PASSPORT = "PASSPORT"


def process_ocr(image_path):
    print("Processing image for OCR...")
    # loop 600 to 800
    image_size = [600, 700, 800]
    regex_found = False

    for size in image_size:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (size,size), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ocr_result = pytesseract.image_to_string(img).upper()
        print(ocr_result)
        # print("=====================================")

        results = ocr_result.split()
        # print(results)
        # print("=====================================")

        # check ic
        for result in results:
            regex_found = re.search(IC_NUMBER_REGREX, result)
            if bool(regex_found):
                break
        if any(pattern in results for pattern in IC_PATTERNS) or bool(regex_found):
            return IDENTITY_CARD

        # check driving, with regrex Date and IC Number
        for result in results:
            regex_found = re.search(DRIVING_DATE_REGREX, result)
            if bool(regex_found):
                break

            regex_found = re.search(DRIVING_IC_NUMBER_REGREX, result)
            if bool(regex_found):
                break
        if any(pattern in results for pattern in DRIVING_PATTERN) or bool(regex_found):
            return DRIVING_LICENSE

        # check passport
        for result in results:
            regex_found = re.search(PASSPORT_DATE_REGREX, result)
            if bool(regex_found):
                break
        if any(pattern in results for pattern in PASSPORT_PATTERNS) or bool(regex_found):
            return PASSPORT

        # cv2.imshow('image', img)
        # cv2.waitKey()