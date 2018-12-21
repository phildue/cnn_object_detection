import numpy as np

from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.Polygon import Polygon


class Encoder:

    def __init__(self, anchor_dims, img_norm, grids, n_polygon=4, iou_min=0.01, iou_ignore=0.7, verbose=False):

        self.iou_ignore = iou_ignore
        self.verbose = verbose
        self.iou_min = iou_min
        self.anchor_dims = anchor_dims
        self.n_polygon = n_polygon
        self.grids = grids
        self.norm = img_norm
        self.unmatched = 0
        self.unmatched_boxes = []
        self.matched = 0
        self.n_boxes = [len(a) for a in anchor_dims]

    @staticmethod
    def generate_encoding(norm, grids, anchor_dims, n_polygon):

        n_output_layers = len(grids)
        anchors = [Encoder.generate_anchor_layer(norm, grids[i], anchor_dims[i], n_polygon) for i in
                   range(n_output_layers)]
        anchors = np.concatenate(anchors, 0)

        return anchors

    @staticmethod
    def generate_anchor_layer(norm, grid, anchor_dims, n_polygon):
        n_boxes = len(anchor_dims)
        anchor_t = np.zeros((grid[0], grid[1], n_boxes, n_polygon + 2)) * np.nan

        cell_height = norm[0] / grid[0]
        cell_width = norm[1] / grid[1]
        cx = np.linspace(0, norm[1] - cell_width, grid[1])
        cy = np.linspace(0, norm[0] - cell_height, grid[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)
        anchor_t[:, :, :, 0] = cx_grid
        anchor_t[:, :, :, 1] = cy_grid
        anchor_t[:, :, :, -2:] = cell_width, cell_height
        for i in range(n_boxes):
            anchor_t[:, :, i, 2:4] = anchor_dims[i]

        anchor_t = np.reshape(anchor_t, (grid[0] * grid[1] * n_boxes, -1))

        return anchor_t

    def _assign_truth(self, label: ImgLabel):

        label_grids = []
        for ig, g in enumerate(self.grids):
            c_w = self.norm[1] / g[1]
            c_h = self.norm[0] / g[0]
            y = np.zeros((g[0], g[1], self.n_boxes[ig], self.n_polygon + 1))
            y[:, :, :, 0] = np.nan
            cx = np.linspace(c_w / 2, self.norm[1] - c_w / 2, g[1])
            cy = np.linspace(c_h / 2, self.norm[0] - c_h / 2, g[0])
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = np.expand_dims(cx_grid, -1)
            cy_grid = np.expand_dims(cy_grid, -1)
            y[:, :, :, 1] = cx_grid
            y[:, :, :, 2] = cy_grid
            for ia in range(self.n_boxes[ig]):
                y[:, :, ia, 3:] = self.anchor_dims[ig][ia]

            label_grids.append(y)

        boxes_true = [o.poly for o in label.objects]

        for bt in boxes_true:
            b = bt.copy()
            iou_max = 0.0
            ig_max = 0
            icx_max = 0
            icy_max = 0
            ia_max = 0
            b.points[:, 1] = self.norm[0] - b.points[:, 1]
            for ig, g in enumerate(self.grids):

                g_w = self.norm[1] / g[1]
                g_h = self.norm[0] / g[0]

                icx = int(np.floor(b.cx / g_w))
                icy = int(np.floor(b.cy / g_h))

                if icx >= g[1] or icx < 0 or 0 > icy or icy >= g[0]:
                    break

                for ia in range(self.n_boxes[ig]):
                    aw, ah = self.anchor_dims[ig][ia]
                    acx = g_w * (icx + 0.5)
                    acy = g_h * (icy + 0.5)
                    anchor = Polygon.from_quad_t_centroid(np.array([[acx, acy, aw, ah]]))
                    iou = b.iou(anchor)

                    if iou > self.iou_ignore:
                        label_grids[ig][icy, icx, ia, 0] = -1.0

                    if iou > iou_max:
                        iou_max = iou
                        ig_max = ig
                        icx_max = icx
                        icy_max = icy
                        ia_max = ia

            if iou_max > self.iou_min:

                label_grids[ig_max][icy_max, icx_max, ia_max] = 1.0, b.cx, b.cy, b.width, b.height
                self.matched += 1
                if self.verbose:
                    print("Assigned Anchor: {}-{}-{}-{}: {}".format(ig_max, icx_max, icy_max, ia_max,
                                                                    label_grids[ig_max][icy_max, icx_max, ia_max]))

            else:
                self.unmatched += 1
                self.unmatched_boxes.append(b)
                if self.unmatched % 500 == 0:
                    # for b in self.unmatched_boxes:
                    #     print("{},".format(b))
                    print("Un/matched boxes: {}/{}".format(self.unmatched, self.matched))
                    self.unmatched_boxes = []

        label_t = []
        for ig, g in enumerate(self.grids):
            label_t.append(np.reshape(label_grids[ig], (g[0] * g[1] * self.n_boxes[ig], self.n_polygon + 1)))

        label_t = np.vstack(label_t)
        label_t[np.isnan(label_t[:, 0]), 0] = 0.0

        if np.any(np.isnan(label_t)) or np.any(np.isinf(label_t)):
            raise ValueError("Invalid Ground Truth")

        return label_t

    def _normalize_label(self, label_t, encoding):
        conf = label_t[:, 0]
        b_cx = label_t[:, 1]
        b_cy = label_t[:, 2]
        b_w = label_t[:, 3]
        b_h = label_t[:, 4]
        xoff = encoding[:, 0]
        yoff = encoding[:, 1]
        p_w = encoding[:, 2]
        p_h = encoding[:, 3]
        cw = encoding[:, 4]
        ch = encoding[:, 5]

        t_cx = (b_cx - xoff) / cw
        t_cy = (b_cy - yoff) / ch

        if np.any(t_cx < 0) or np.any(t_cx > 1) or np.any(t_cy < 0) or np.any(t_cy > 1):
            raise ValueError('Invalid Assignment')

        t_w = b_w / p_w
        t_h = b_h / p_h
        return np.column_stack((conf, t_cx, t_cy, t_w, t_h, xoff, yoff, p_w, p_h, cw, ch))

    def encode_img(self, image: Image):
        img = normalize(image)
        return np.expand_dims(img.array, axis=0)

    def encode_label(self, label: ImgLabel):
        """
        Encodes bounding box in ground truth tensor.

        :param label: image label containing objects, their bounding boxes and names
        :return: label-tensor

        """
        encoding = Encoder.generate_encoding(self.norm, self.grids, self.anchor_dims, self.n_polygon)
        y = self._assign_truth(label)
        label_t = self._normalize_label(y, encoding)
        return label_t

    @staticmethod
    def logit(c):
        return np.log(c / (1 - c))

    def encode_img_batch(self, images: [Image]) -> np.array:
        imgs_enc = []
        for img in images:
            img_t = self.encode_img(img)
            imgs_enc.append(img_t)
        img_t = np.concatenate(imgs_enc, 0)
        return img_t

    def encode_label_batch(self, labels: [ImgLabel]) -> np.array:
        labels_enc = []
        for label in labels:
            label_t = self.encode_label(label)
            label_t = np.expand_dims(label_t, 0)
            labels_enc.append(label_t)
        label_t = np.concatenate(labels_enc, 0)
        return label_t
