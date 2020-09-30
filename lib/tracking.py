import cv2
import numpy as np

TRACKER_TYPE_LIST = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createOneTracker(type):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    tracker_type = type
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    return tracker


def createMultiTracker():
    trackers = cv2.MultiTracker_create()
    return trackers


def _calManhattanDistance(point_1, point_2):
    vector1 = np.mat(list(point_1))
    vector2 = np.mat(list(point_2))
    re = np.sum(np.abs(vector1-vector2))
    return re


def cost(predict_points, track_points):
    distance_head = _calManhattanDistance(predict_points[0], track_points[0])
    distance_tail = _calManhattanDistance(predict_points[1], track_points[1])
    a = (distance_head + distance_tail)
    #b = 2.0 * _cal_manhattan_distance(predict_points[0], predict_points[1]) * _cal_manhattan_distance(track_points[0], track_points[1])
    b = _calManhattanDistance(predict_points[0], predict_points[1])# * predict_points[0] * predict_points[1]
    return 0.5 * a / b
