import cv2
import numpy as np


def resize():
    pass

def gaussian_and_otsu(img_obj):
    blur = cv2.GaussianBlur(img_obj, (7,7), 0)
    ret, th = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return th


def equalize_hist(img_obj):
    lab_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    equa = cv2.equalizeHist(l)

    equalized_lab_img = cv2.merge((equa, a, b))
    equalized_img = cv2.cvtColor(equalized_lab_img, cv2.COLOR_LAB2BGR)

    return equalized_img


def CLAHE(img_obj):
    lab_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)

    clahe_lab_img = cv2.merge((clahe_img, a, b))
    clahe_img = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2BGR)

    return clahe_img


def get_gray_img(img_obj):
    return cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)


def adaptive_gaussian_threshold(img_obj):
    blur = cv2.GaussianBlur(img_obj, (7, 7), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 8)


def invert_img(img_obj):
    return cv2.bitwise_not(img_obj)


def median_blur(img_obj):
    return cv2.medianBlur(img_obj, 7)


def get_contours(th):
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(th, rect_kern, iterations=1)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
