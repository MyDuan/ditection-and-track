import cv2
from lib.img_processing import *


if __name__ == '__main__':

    source = './sample_data/pig_sample_video.mp4'
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./sample_data/find_multi_pig.mp4', fourcc, 20.0, (1500,1520))

    y_start = 620
    y_end = 2120

    while True:
        ret, frame = cap.read()
        output_frame, key_points = imgProcessing(frame, y_start, y_end)
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

