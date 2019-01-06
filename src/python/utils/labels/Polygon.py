import copy

import numpy as np


class Polygon:

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_quad_t_centroid(coord_t):
        n_boxes = coord_t.shape[0]
        boxes = []
        for i in range(n_boxes):
            cx, cy, w, h = coord_t[i]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy - h / 2
            x3 = cx + w / 2
            y3 = cy + h / 2
            x4 = cx - w / 2
            y4 = cy + h / 2
            b = Polygon(np.array([[x1, y1],
                                  [x2, y2],
                                  [x3, y3],
                                  [x4, y4]]))
            boxes.append(b)

        if len(boxes) > 1:
            return boxes
        else:
            return boxes[0]

    @staticmethod
    def from_quad_t_minmax(coord_t):
        n_boxes = coord_t.shape[0]
        boxes = []
        for i in range(n_boxes):
            x_min, y_min, x_max, y_max = coord_t[i, :]
            b = Polygon(np.array([[x_min, y_min],
                                  [x_max, y_min],
                                  [x_max, y_max],
                                  [x_min, y_max]]))
            boxes.append(b)

        if len(boxes) > 1:
            return boxes
        else:
            return boxes[0]

    def __init__(self, points: np.array):

        self.points = points

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.area + box.area - intersection
        if union == 0:
            return 1
        else:
            return intersection / union

    def intersect(self, box):
        width = self._overlap([self.x_min, self.x_max], [box.x_min, box.x_max])
        height = self._overlap([self.y_min, self.y_max], [box.y_min, box.y_max])
        return width * height

    @staticmethod
    def _overlap(interval_a, interval_b):
        a, b = interval_a
        c, d = interval_b
        if c < a:
            if d < a:
                return 0
            else:
                return min(b, d) - a
        else:
            if b < c:
                return 0
            else:
                return min(b, d) - c

    @property
    def area(self):
        return self.width * self.height

    @property
    def to_quad_t_minmax(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    @property
    def to_quad_t_centroid(self):
        return np.array([self.cx, self.cy, self.width, self.height])

    @property
    def x_min(self):
        return np.min(self.points[:, 0])

    @property
    def x_max(self):
        return np.max(self.points[:, 0])

    @property
    def y_min(self):
        return np.min(self.points[:, 1])

    @property
    def y_max(self):
        return np.max(self.points[:, 1])

    @property
    def cx(self):
        return self.x_min + self.width / 2

    @property
    def cy(self):
        return self.y_min + self.height / 2

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def aspect_ratio(self):
        return self.height / self.width

    def __repr__(self):
        return '[({0:.2f},{1:.2f}) --> ({2:.2f},{3:.2f})]'.format(self.x_min, self.y_min, self.x_max,
                                                                  self.y_max)
