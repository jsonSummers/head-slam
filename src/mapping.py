import numpy as np

class MapPoint:
    def __init__(self, point3d):
        self.point3d = point3d
        self.observations = []

    def add_observation(self, frame_id, keypoint_id):
        self.observations.append((frame_id, keypoint_id))

class Map:
    def __init__(self):
        self.points = []

    def add_point(self, point3d):
        self.points.append(MapPoint(point3d))

    def add_observation(self, point_id, frame_id, keypoint_id):
        self.points[point_id].add_observation(frame_id, keypoint_id)

def triangulate_points(keypoints1, keypoints2, matches, camera_matrix, rotation_matrix1, tvec1, rotation_matrix2, tvec2):
    #testing:
    print("Length of keypoints1:", len(keypoints1))
    print("Length of matches:", len(matches))
    for match in matches:
        if match.queryIdx >= len(keypoints1):
            print("Invalid match.queryIdx:", match.queryIdx)


    # Convert matches to homogeneous coordinates
    points1 = np.array([keypoints1[match.queryIdx].pt + (1,) for match in matches])
    points2 = np.array([keypoints2[match.trainIdx].pt + (1,) for match in matches])

    # Construct the projection matrices
    projection_matrix1 = np.dot(camera_matrix, np.hstack((rotation_matrix1, tvec1)))
    projection_matrix2 = np.dot(camera_matrix, np.hstack((rotation_matrix2, tvec2)))

    # Triangulate the 3D points
    points4d = cv2.triangulatePoints(projection_matrix1, projection_matrix2, points1.T, points2.T)
    points3d = (points4d[:3] / points4d[3]).T  # Transpose for correct shape

    return points3d
