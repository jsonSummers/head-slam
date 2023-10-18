import numpy as np
import cv2


class MapPoint:
    def __init__(self, point3d):
        self.point3d = point3d
        self.observations = []

    def add_observation(self, frame_id, keypoint_id):
        self.observations.append((frame_id, keypoint_id))

    # def correct_point(self):


class Map:
    def __init__(self):
        self.points = []

    def add_point(self, point3d):
        self.points.append(MapPoint(point3d))

    def add_observation(self, point_id, frame_id, keypoint_id):
        self.points[point_id].add_observation(frame_id, keypoint_id)