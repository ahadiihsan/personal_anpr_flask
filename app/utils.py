from dateutil import parser
from datetime import datetime
import time
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy import ndimage
from typing import Tuple, Union
import math
import imutils as im
import Levenshtein
from app import save_image

# load denoiser model
denoiser = load_model("./models/denoiser_ae2.h5")


def convert_iso_to_timestamp(date):
    return int(time.mktime(parser.isoparse(date).timetuple()))


def convert_timestamp_to_iso(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%S')


# noinspection SpellCheckingInspection
def get_tesseract_config(pytesseract):
    # for Linux it's usually available trough environment path
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = (
            r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        )

    # we could also use user-pattern like \A\A \d\d\d-\A\A
    config = (
            """
            --psm 11
            -c tessedit_char_whitelist=123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-
            """
        )
    return config


def letterbox(
            img,
            new_shape=(512, 512),
            color=(114, 114, 114),
            auto=True,
            scaleFill=False,
            scaleup=True
        ):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

        # width, height ratios
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
    return img, ratio, (dw, dh)


def line_remover(plate):
    edges = cv2.Canny(plate, 50, 150, apertureSize=3)
    if save_image: cv2.imwrite("./image/img_canny.jpeg", edges)
    height, width = edges.shape[:2]
    edges = edges[0: int(height/2), 0:width]
    minLineLength = 0
    maxLineGap = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength, maxLineGap)

    if lines is None:
        return True
    else:
        return False


def asserter(plate, cut_index=0):
    if line_remover(plate):
        cut_index = cut_index
    else:
        cut_index += 3
    return cut_index


def boundary(plate):
    cut_index = 0
    height, width = plate.shape[:2]
    cut_index = asserter(plate, cut_index)
    plate = plate[cut_index:height, 0: width]
    return plate


def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(
            imgGrayscale, cv2.MORPH_TOPHAT, structuringElement
        )
    imgBlackHat = cv2.morphologyEx(
            imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement
        )

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    if save_image: cv2.imwrite(
            "./image/img_imgGrayscalePlusTopHat.jpeg", imgGrayscalePlusTopHat
        )
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(
            imgGrayscalePlusTopHat, imgBlackHat
        )
    if save_image: cv2.imwrite(
            "./image/img_imgGrayscalePlusTopHatMinusBlackHat.jpeg",
            imgGrayscalePlusTopHatMinusBlackHat
        )
    return imgGrayscalePlusTopHatMinusBlackHat


def unsharp_mask(
            image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0
        ):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img


def image_enhancement(img):
    img = modcrop(img, 3)
    # img = cv2.resize(img, dsize=(256, 80), interpolation=cv2.INTER_CUBIC)
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    pre = denoiser.predict(Y)

    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    temp = shave(temp, 6)
    # temp = cv2.resize(temp, dsize=(256, 80), interpolation=cv2.INTER_CUBIC)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    return output


def is_horizontal(lines):
    for x1, y1, x2, y2 in lines[0]:
        return (x2-x1 == 0 or y2-y1 == 0)


def find_line(lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            if not (x2-x1 == 0 or y2-y1 == 0):
                return [[x1, y1, x2, y2]]
            else:
                return [[x1, y1, x2, y2+0.5]]


def rotate(image):
    edges = cv2.Canny(image, 50, 150)
    if save_image: cv2.imwrite("./image/img_canned.jpeg", edges)
    lines = cv2.HoughLinesP(
            edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=3
        )
    # if save_image: cv2.imwrite("./image/img_lines1.jpeg", lines)

    if lines is None or is_horizontal(lines):
        lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 70, minLineLength=70, maxLineGap=3
            )
        # if save_image: cv2.imwrite("./image/img_lines2.jpeg", lines)
        if lines is None or is_horizontal(lines):
            lines = cv2.HoughLinesP(
                    edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=3
                )
            # if save_image: cv2.imwrite("./image/img_lines3.jpeg", lines)

    if lines is None:
        return image

    newline = find_line(lines)
    angles = []

    for x1, y1, x2, y2 in newline:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    rotated = ndimage.rotate(image, median_angle)

    return rotated


def rotate_image_by_angel(
            image, angle, background: Union[int, Tuple[int, int, int]]
        ):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = (
                abs(np.sin(angle_radian) * old_height) +
                abs(np.cos(angle_radian) * old_width)
            )
    height = (
                abs(np.sin(angle_radian) * old_width) +
                abs(np.cos(angle_radian) * old_height)
             )
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(height)), int(round(width))),
            borderValue=background
        )


def detect_corners_from_contour(img, cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    return np.array(approx_corners)


def order_points(pts):
    # source pyimagesearch
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # source pyimagesearch
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # dst = np.float(
    #         [[0, 0], [0, maxHeight - 1],
    #         [maxWidth - 1, 0],
    #         [maxWidth - 1, maxHeight - 1]]
    #     )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(
                                            img,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE
                                        )
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 3000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) >= 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    warped = orig  # Stores the warped license plate image
    if index is not None:  # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32)  # Source points
        warped = four_point_transform(orig, src)

        # Menghitung sudut maksimum dari gambar
        degree = count_angle(order_points(src))
        epsilon = 1

        # Error Correction untuk gambar miring dengan sudut kecil
        if abs(degree) < 5:
            warped = im.rotate_bound(warped, 2)
        else:
            # Error Correction untuk gambar yang miring
            if degree <= 0:
                # Jika sudut negatif (sisi kiri lebih rendah dari sisi kanan)
                warped = im.rotate_bound(warped, -1*(0.1*degree) + epsilon)
            else:
                # Jika sudut positif (sisi kanan lebih rendah dari sisi kiri)
                warped = im.rotate_bound(warped, 0.05*degree + epsilon)

    return biggest, imgContour, warped  # Change - also return drawn image


def pascal_voc_to_coco(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    return [x1, y1, x2 - x1, y2 - y1]


# Fungsi untuk menghitung sudut maksimal garis horizontal dari
# Edges plat yang terdeteksi
def count_angle(points):
    atas_kiri, atas_kanan, bawah_kanan, bawah_kiri = points

    # Menghitung lebar dan tinggi di sepasang sisi
    height_1 = (atas_kanan[1]-atas_kiri[1])
    width_1 = (atas_kanan[0]-atas_kiri[0])
    height_2 = (bawah_kanan[1]-bawah_kiri[1])
    width_2 = (bawah_kanan[0]-bawah_kiri[0])

    # Mencari sudut maksimum relatif terhadap garis horizontal
    # Sudut positif arah jarum jam dari sumbu x positif
    degree_1 = math.atan2(height_1, width_1)
    degree_2 = math.atan2(height_2, width_2)
    if degree_1 >= 0 and degree_2 >= 0:
        degree = max(degree_1, degree_2)
    elif degree_1 < 0 and degree_2 >= 0:
        degree = degree_1
    elif degree_1 >= 0 and degree_2 < 0:
        degree = degree_2
    elif degree_1 < 0 and degree_2 < 0:
        degree = -1*(max(abs(degree_1), abs(degree_2)))
    else:
        degree = 0

    # mengubah dari unit radian ke derajat
    pi = math.pi
    degree_in_degree = (degree/(2*pi)*360)

    # return sudut dalam satuan derajat
    return degree_in_degree


def calculate_similarity(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    max_length = max(len(str1), len(str2))
    similarity = 1 - (distance / max_length)
    return similarity