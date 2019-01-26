import numpy as np

from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon


class YoloV3Decoder:
    def __init__(self,
                 img_size,
                 grids,
                 anchor_dims,
                 n_classes,
                 n_polygon=4):
        self.n_classes = n_classes
        self.n_polygon = n_polygon
        self.grids = grids
        self.img_size = img_size
        self.n_boxes = [len(a) for a in anchor_dims]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def decode_netout(self, y):
        """
        Convert label tensor to objects of type Box.
        :param y: (#total boxes,[objectness,class probabilities one hot encoded, box cx, box cy, box width, box height,
                                anchor with, anchor height, grid cx, grid cy, cell width, cell height])
                                label tensor as predicted by network
        :return: boxes
        """
        conf_t = self.sigmoid(y[:, 0])
        class_t = self.softmax(y[:, 1:self.n_classes+1])
        coord_t = y[:, self.n_classes + 1:self.n_classes + 1 + self.n_polygon]
        enc_t = y[:, self.n_classes + 1 + self.n_polygon:]

        coord_t_dec = self.decode_coord(coord_t, enc_t)

        boxes = Polygon.from_quad_t_centroid(coord_t_dec)

        labels = []
        for i, b in enumerate(boxes):
            conf = conf_t[i]*np.max(class_t[i])
            class_id = np.argmax(class_t[i, :])
            label = ObjectLabel(class_id, conf, b)
            labels.append(label)

        return ImgLabel(labels)

    def decode_coord(self, coord_t, enc_t):
        """
        Decode the coordinates of the bounding boxes from the label tensor to absolute coordinates in the image.
        :param enc_t: (#total boxes, [x offset, y offset, anchor w, anchor h, cell w, cell h])
        :param coord_t: (#total boxes, [predicted dcx, predicted dcy, predicted dw, predicted dh])
        :return: label: (#total boxes, [b_cx, b_cy, b_w, b_h]) tensor in image coordinates
        """

        t_cx = coord_t[:, 0]
        t_cy = coord_t[:, 1]
        t_w = coord_t[:, 2]
        t_h = coord_t[:, 3]

        x_off = enc_t[:, 0]
        y_off = enc_t[:, 1]
        p_w = enc_t[:, 2]
        p_h = enc_t[:, 3]
        cw = enc_t[:, 4]
        ch = enc_t[:, 5]

        b_cx = self.sigmoid(t_cx) * cw + x_off
        b_cy = self.sigmoid(t_cy) * ch + y_off
        b_w = self.exp(t_w) * p_w
        b_h = self.exp(t_h) * p_h

        b_cy = self.img_size[0] - b_cy

        return np.column_stack([b_cx, b_cy, b_w, b_h])

    def decode_netout_batch(self, netout) -> [ImgLabel]:
        labels = []
        for i in range(netout.shape[0]):
            label = self.decode_netout(netout[i])
            labels.append(label)

        return labels
