import cv2
import numpy as np


def estimate_camera_pose(keypoints2, matches, camera, triangulated_points):
    # Convert keypoints to format needed by solvePnP
    obj_points = []  # 3D world points
    img_points = []  # Corresponding 2D image points

    # for match in filtered_matches:
    for n in range(len(matches)):
        # Get the index of the corresponding triangulated point
        obj_points.append(triangulated_points[n])
        img_points.append(keypoints2[n].pt)

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    # Provide an initial guess for camera pose based on the previous frame
    initial_rvec, initial_tvec = camera.get_previous_pose()

    # retval, rvec, tvec = cv2.solvePnPRansac(obj_points, img_points, camera.camera_matrix, None, rvec=initial_rvec,
    #                                  tvec=initial_tvec, flags=cv2.SOLVEPNP_ITERATIVE)

    # Use solvePnP to estimate camera pose
    retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera.camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

    if retval:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Update the camera's pose
        camera.update_pose(rotation_matrix, tvec)

        # Optionally, you can use inliers to filter your obj_points and img_points for further processing if needed.

        return rotation_matrix, tvec
    else:
        print("Failed to estimate camera pose.")
        return None, None
