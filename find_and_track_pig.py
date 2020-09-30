import cv2
from munkres import Munkres
from lib.img_processing import *
from lib.tracking import *


if __name__ == '__main__':

    source = './sample_data/pig_sample_video.mp4'
    cap = cv2.VideoCapture(source)
    ret, frame_first = cap.read()

    y_start = 620
    y_end = 2120

    _, key_points = imgProcessing(frame_first, y_start, y_end)
    roi_of_img = frame_first[:, y_start:y_end]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./sample_data/multi_pig_track_with_pairing.mp4',
                          fourcc, 20.0, (y_end - y_start, frame_first.shape[0]))

    tracker_type = TRACKER_TYPE_LIST[2]
    trackers = createMultiTracker()
    for item in key_points:
        tracker = createOneTracker(tracker_type)
        trackers.add(tracker, roi_of_img, (min(item[1][0], item[0][0]), min(item[1][1], item[0][1]),
                                           abs(item[0][0] - item[1][0]),
                                           abs(item[0][1] - item[1][1])))
    count = 1
    while True:
        ret, frame = cap.read()
        timer = cv2.getTickCount()
        roi_of_img = frame[:, y_start:y_end]
        ok, boxes = trackers.update(roi_of_img)
        # FPSを表示する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(roi_of_img, "FPS : " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        if ok:
            for index, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(roi_of_img, p1, p2, (0, 255, 0))

                roi_of_img = drawTextBox(roi_of_img, newbox[0] + newbox[2] / 2, newbox[1] + newbox[3] / 2,
                                         display_str_list=('No.' + str(index)))
        else:
            print(ok)
            cv2.putText(roi_of_img, "Some pig can not be track", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)
            if count % 3 == 0:
                trackers = createMultiTracker()
                _, key_points = imgProcessing(frame, y_start, y_end)
                m = Munkres()
                cost_mat = []
                for item in boxes:
                    cost_row = []
                    for key_point in key_points:
                        boxe_point = [(item[0]+item[2], item[1]+item[3]),
                                      (item[0], item[1])]
                        cost_row.append(cost(boxe_point, key_point))
                    cost_mat.append(cost_row)

                re = m.compute(cost_mat)
                unpredict_list = list(range(len(boxes)))
                for i in range(len(key_points)):
                    tracker = createOneTracker(tracker_type)
                    trackers.add(tracker, roi_of_img, (min(key_points[re[i][1]][1][0], key_points[re[i][1]][0][0]),
                                                       min(key_points[re[i][1]][1][1], key_points[re[i][1]][0][1]),
                                                       abs(key_points[re[i][1]][0][0] - key_points[re[i][1]][1][0]),
                                                       abs(key_points[re[i][1]][0][1] - key_points[re[i][1]][1][1])))
                    p1 = (int(key_points[re[i][1]][1][0]), int(key_points[re[i][1]][1][1]))
                    p2 = (int(key_points[re[i][1]][0][0]), int(key_points[re[i][1]][0][1]))
                    cv2.rectangle(roi_of_img, p1, p2, (0, 255, 0), 2, 1)

                    ROIImg = drawTextBox(roi_of_img,
                                         min(key_points[re[i][1]][1][0], key_points[re[i][1]][0][0]) +
                                         abs(key_points[re[i][1]][1][0] - key_points[re[i][1]][0][0]) / 2,
                                         min(key_points[re[i][1]][1][1], key_points[re[i][1]][0][1]) +
                                         abs(key_points[re[i][1]][1][1] - key_points[re[i][1]][0][1]) / 2,
                                         display_str_list=('No.' + str(re[i][1])))
                    unpredict_list.remove(re[i][0])
                if len(boxes) != len(key_points):
                    for index in unpredict_list:
                        trackers.add(tracker, roi_of_img, tuple(boxes[index]))
                        p1 = (int(boxes[index][0]), int(boxes[index][1]))
                        p2 = (int(boxes[index][0] + boxes[index][2]), int(boxes[index][1] + boxes[index][3]))
                        cv2.rectangle(roi_of_img, p1, p2, (0, 0, 255))
                        roi_of_img = drawTextBox(roi_of_img, boxes[index][0] + boxes[index][2] / 2,
                                                 boxes[index][1] + boxes[index][3] / 2,
                                                 display_str_list=('No.' + str(index)))

        out.write(roi_of_img)
        cv2.imshow("Tracking", roi_of_img)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

