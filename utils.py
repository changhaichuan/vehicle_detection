import cv2
import numpy as np


def get_ptm():

    pts_1 = np.float32([[704, 353], [728, 396], [553, 1031], [684, 925], [872, 766], [1131, 502], [1626, 595],
                        [1294, 636], [1264, 680], [1155, 835], [1105, 904], [1662, 915], [1846, 1016]])

    pts_2 = np.float32([[1167, 541], [1042, 518], [506, 261], [528, 305], [566, 393], [688, 602], [446, 713],
                        [531, 570], [514, 534], [473, 447], [455, 413], [321, 540], [241, 544]])

    matrix = cv2.findHomography(pts_1, pts_2, method=cv2.RANSAC)

    return matrix[0]


# BGR values
vehicle_colours = {
    'motorcycle': (0, 255, 255),
    'car': (0, 255, 0),
    'bus': (255, 0, 0),
    'truck': (0, 0, 255)
}

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


