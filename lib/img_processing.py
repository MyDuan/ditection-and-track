import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def drawTextBox(image, x, y, color='red', thickness=4, display_str_list=()):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # font = ImageFont.load_default()
    font = ImageFont.truetype("simsun.ttc", 24, encoding="utf-8")
    display_str_width = [font.getsize(ds)[0] for ds in display_str_list]
    display_str_height = [font.getsize(ds)[1] for ds in display_str_list]

    total_display_str_width = sum(display_str_width) + max(display_str_width) * 1.1
    total_display_str_height = max(display_str_height)

    text_bottom = y + total_display_str_height / 2.0

    text_right = x + total_display_str_width / 2.0

    draw.rectangle(
        [(x - total_display_str_width / 2.0, text_bottom), (text_right, text_bottom - total_display_str_height)],
        fill=color)

    for index in range(len(display_str_list[::1])):
        current_right = (
                    x - total_display_str_width / 2.0 + (max(display_str_width)) + sum(display_str_width[0:index + 1]))

        if current_right < text_right:
            display_str = display_str_list[:index + 1]
        else:
            display_str = display_str_list[0:index - 1] + '...'
            break

    draw.text(
        (x - total_display_str_width / 2.0 + max(display_str_width) / 2, text_bottom - total_display_str_height),
        display_str,
        fill='black',
        font=font)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _watershedProcessing(crop_img):
    # pre processing
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    gray = clahe.apply(gray)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

    # get sure background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(mb, kernel, iterations=3)

    # get sure foreground
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    ret, surface = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), dtype=np.uint8)
    dist = cv2.dilate(surface, kernel)
    ret, surface = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), dtype=np.uint8)
    cv2.dilate(surface, kernel)
    surface_fg = np.uint8(surface)

    # get unknown area (unknown area in background and foreground)
    unknown = cv2.subtract(sure_bg, surface_fg)

    # label the area
    ret, markers = cv2.connectedComponents(surface_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # watershed
    markers = cv2.watershed(crop_img, markers=markers)
    return markers


def _detectionProcessing(crop_img, markers, label):
    mask = np.zeros(crop_img[:, :, 0].shape, dtype="uint8")
    mask[markers == label] = 255
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def _fitAndMark(scr_output, cnt, pig_num):
    ellipse = cv2.fitEllipse(cnt)
    b = ellipse[1][0]
    a = ellipse[1][1]
    c_x = ellipse[0][0]
    c_y = ellipse[0][1]
    theta = ellipse[2]
    if theta > 90:
        theta = theta - 90
    else:
        theta = theta + 90

    if 500 > a > 180 and 60 < b < 250:
        scr_output = cv2.ellipse(scr_output, ellipse, (0, 255, 255), 2)
        rmajor = max(a, b) / 3
        x_head = c_x + math.cos(math.radians(theta)) * rmajor
        y_head = c_y + math.sin(math.radians(theta)) * rmajor
        x_tail = c_x + math.cos(math.radians(theta + 180)) * rmajor
        y_tail = c_y + math.sin(math.radians(theta + 180)) * rmajor
        cv2.line(scr_output, (int(x_head), int(y_head)), (int(x_tail), int(y_tail)), (0, 0, 255), 3)
        scr_output = drawTextBox(scr_output, ellipse[0][0], ellipse[0][1],
                                 display_str_list=('No.' + str(pig_num)))
        key_point = [(x_head, y_head), (x_tail, y_tail)]
        return key_point, scr_output
    else:
        return None, scr_output


def imgProcessing(input_frame, y_start, y_end, num_of_pigs=16):
    src = input_frame.copy()
    crop_img = src[:, y_start:y_end]
    markers = _watershedProcessing(crop_img)
    scr_output = crop_img.copy()
    #scr_output[markers == -1] = (0, 0, 255)

    pig_num = 0
    key_points = []

    for label in np.unique(markers):
        if label <= 1:
            continue
        contours = _detectionProcessing(crop_img, markers, label)
        for n, cnt in enumerate(contours):
            if pig_num >= num_of_pigs:
                break
            if len(cnt) > 4:
                key_point, scr_output = _fitAndMark(scr_output, cnt, pig_num)
                if key_point is not None:
                    key_points.append(key_point)
                    pig_num += 1
    return scr_output, key_points
