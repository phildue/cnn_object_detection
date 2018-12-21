import numpy as np

from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon


class Decoder:
    def __init__(self,
                 norm,
                 grid,
                 anchor_dims,
                 n_polygon=4):
        self.n_polygon = n_polygon
        self.grid = grid
        self.norm = norm
        self.n_boxes = [len(a) for a in anchor_dims]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def decode_netout(self, label_t):
        """
        Convert label tensor to objects of type Box.
        :param label_t: y as fed for learning
        :return: boxes
        """
        coord_t = label_t[:, 1:]
        class_t = self.sigmoid(label_t[:, :1])
        coord_t_dec = self.decode_coord(coord_t)
        # coord_t_dec = np.reshape(coord_t_dec, (-1, self.n_polygon))
        # class_t = np.reshape(class_t, (-1, 1))
        boxes = Polygon.from_quad_t_centroid(coord_t_dec)

        labels = []
        for i, b in enumerate(boxes):
            conf = np.max(class_t[i])
            class_id = np.argmax(class_t[i, :])
            label = ObjectLabel(ObjectLabel.id_to_name(class_id), conf, b)
            labels.append(label)

        return ImgLabel(labels)

    def decode_coord(self, coord_t):
        """
        Decode the coordinates of the bounding boxes from the label tensor to absolute coordinates in the image.
        :param coord_t: label tensor (only coordinates)
        :return: label tensor in absolute coordinates
        """

        t_cx = coord_t[:, 0]
        t_cy = coord_t[:, 1]
        t_w = coord_t[:, 2]
        t_h = coord_t[:, 3]
        xoff = coord_t[:, 4]
        yoff = coord_t[:, 5]
        p_w = coord_t[:, 6]
        p_h = coord_t[:, 7]
        cw = coord_t[:, 8]
        ch = coord_t[:, 9]

        b_cx = self.sigmoid(t_cx) * cw + xoff
        b_cy = self.sigmoid(t_cy) * ch + yoff
        b_w = np.exp(t_w) * p_w
        b_h = np.exp(t_h) * p_h

        b_cy = self.norm[0] - b_cy

        return np.column_stack([b_cx, b_cy, b_w, b_h])

    def decode_netout_batch(self, netout) -> [ImgLabel]:
        labels = []
        for i in range(netout.shape[0]):
            label = self.decode_netout(netout[i])
            labels.append(label)

        return labels