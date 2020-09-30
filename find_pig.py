import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont


def draw_text_box_on_image(image,
                           x,
                           y,
                           color='red',
                           thickness=4,
                           display_str_list=()):
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


def img_processing(input_frame, num_of_pigs=16):
    src = input_frame.copy()
    c = 620  # y start
    d = 2120  # y end
    cropImg = src[:, c:d]
    #blurred = cv2.pyrMeanShiftFiltering(cropImg, 10, 100)
    gray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    gray = clahe.apply(gray)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    ret, surface = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    kernel1 = np.ones((2, 2), dtype=np.uint8)
    dist = cv2.dilate(surface, kernel1)
    ret, surface = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    kernel1 = np.ones((2, 2), dtype=np.uint8)
    cv2.dilate(surface, kernel1)
    surface_fg = np.uint8(surface)
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cropImg, markers=markers)
    scr_output = cropImg.copy()
    #scr_output[markers == -1] = (0, 0, 255)
    pig_num = 0
    back_area = {}
    length = {}
    weight = {}
    key_points = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for n, cnt in enumerate(contours):
            if pig_num >= num_of_pigs:
                break
            if len(cnt) > 4:
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


                if a > 180 and a < 500 and b < 250 and b > 60:
                    scr_output = cv2.ellipse(scr_output, ellipse, (0, 255, 255), 2)
                    rmajor = max(a, b) / 3
                    x_head = c_x + math.cos(math.radians(theta)) * rmajor
                    y_head = c_y + math.sin(math.radians(theta)) * rmajor
                    x_tail = c_x + math.cos(math.radians(theta + 180)) * rmajor
                    y_tail = c_y + math.sin(math.radians(theta + 180)) * rmajor
                    cv2.line(scr_output, (int(x_head), int(y_head)), (int(x_tail), int(y_tail)), (0, 0, 255), 3)
                    key_points.append([(x_head, y_head), (x_tail, y_tail)])
                    scr_output = draw_text_box_on_image(scr_output, ellipse[0][0], ellipse[0][1],
                                                        display_str_list=('No.' + str(pig_num)))
                    back_area['No.' + str(pig_num)] = round(math.pi * a * b / 4.0, 2)
                    length['No.' + str(pig_num)] = round(a, 2)
                    weight['No.' + str(pig_num)] = 0
                    pig_num += 1
    return scr_output, key_points


if __name__ == '__main__' :

    source = './sample_data/14_nursery_medium_activity_day.mp4'
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./sample_data/find_multi_pig.mp4',fourcc, 20.0, (1500,1520))
    while True:
        ret, frame = cap.read()
        output_frame, key_points = img_processing(frame)
        timer = cv2.getTickCount()
        # FPSを表示する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(output_frame, "FPS : " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        out.write(output_frame)
        cv2.imshow("Finding", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

