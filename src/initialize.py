import numpy as np

def initialize(loader, cloud_map, camera, frames=3):
    print('init')
    get_initial_pose(loader, camera)


def get_initial_pose(loader, camera):
    init_rotation_matrix = np.eye(3)  # Initialize with identity matrix
    init_translation_vector = np.zeros((3, 1))  # Initialize with zeros
    camera.update_pose(init_rotation_matrix, init_translation_vector)
