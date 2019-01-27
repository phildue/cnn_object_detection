import numpy as np

from utils.labels.ImgLabel import ImgLabel
from utils.labels.Polygon import Polygon


class YoloV3Encoder:

    def __init__(self, anchor_dims, img_size, grids, n_classes, n_polygon=4, iou_min=0.01, iou_ignore=0.7,
                 verbose=False):
        """
        Encodes bounding box label to target tensor.

        :param anchor_dims: (#output layers,#anchor boxes,#dimensions) Anchor box dimensions for each output layer
        :param img_size: (height,width) input image dimensions
        :param grids: (#output layers,#grid dimensions) Grid size for each output layer
        :param n_classes: number of classes
        :param n_polygon: number of corners in output anchor
        :param iou_min: minimum overlap to assign responsible anchor
        :param iou_ignore: ignore threshold. predictions of anchors that are not responsible
                           but overlap more than iou_ignore do not incur any loss
        :param verbose: print status message during encoding (0 = none, 1 = modest, 2 = details)

        """
        self.ignored = 0
        self.n_classes = n_classes
        self.iou_ignore = iou_ignore
        self.verbose = verbose
        self.iou_min = iou_min
        self.anchor_dims = anchor_dims
        self.n_polygon = n_polygon
        self.grids = grids
        self.img_size = img_size
        self.unmatched = 0
        self.unmatched_boxes = []
        self.matched = 0
        self.n_boxes = [len(a) for a in anchor_dims]

    @staticmethod
    def generate_encoding_tensor(img_size, grids, anchor_dims, n_polygon):
        """
        Generate the tensor with coordinate encoding information.
        :param img_size: (height,width) input image dimensions
        :param grids: (#output layers,#grid dimensions) Grid size for each output layer
        :param anchor_dims: (#anchor boxes,#dimensions) Anchor box dimensions for output layer
        :param n_polygon: number of corners in output anchor

        :return: (#total boxes,[b_cx,b_cy,b_w,b_h,cell_width,cell_height])
                 encoding tensor for all output layers

        """
        n_output_layers = len(grids)
        encoding_t = []
        for i in range(n_output_layers):
            n_boxes = len(anchor_dims[i])
            encoding_layer_t = YoloV3Encoder.generate_encoding_tensor_layer(img_size, grids[i], anchor_dims[i],
                                                                            n_polygon)
            encoding_layer_t = np.reshape(encoding_layer_t, (grids[i][0] * grids[i][1] * n_boxes, -1))
            encoding_t.append(encoding_layer_t)
        encoding_t = np.concatenate(encoding_t, 0)

        return encoding_t

    @staticmethod
    def generate_encoding_tensor_layer(img_size, grid, anchor_dims, n_polygon):
        """
        Generate the tensor with coordinate encoding information.
        :param img_size: (height,width) input image dimensions
        :param grid: (n_grid_vertical,n_grid_horizontal) grid dimensions
        :param anchor_dims: (#anchor boxes,#dimensions) Anchor box dimensions for output layer
        :param n_polygon: number of corners in output anchor

        :return: (#grid_vertical,#grid_horizontal,#boxes,[b_cx,b_cy,b_w,b_h,cell_width,cell_height])
                 encoding tensor for particular output layer

        """
        n_boxes = len(anchor_dims)
        anchor_t = np.zeros((grid[0], grid[1], n_boxes, n_polygon + 2)) * np.nan

        cell_height = img_size[0] / grid[0]
        cell_width = img_size[1] / grid[1]
        cx = np.linspace(0, img_size[1] - cell_width, grid[1])
        cy = np.linspace(0, img_size[0] - cell_height, grid[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)
        anchor_t[:, :, :, 0] = cx_grid
        anchor_t[:, :, :, 1] = cy_grid
        anchor_t[:, :, :, -2:] = cell_width, cell_height
        for i in range(n_boxes):
            anchor_t[:, :, i, 2:4] = anchor_dims[i]

        return anchor_t

    def _create_empty_target_ts(self):
        """
        Create empty target tensors based on hyperparameters
        :return: (#total boxes,#1 + n classes + n polygon + 6])
        """
        class_ts = []
        anchor_ts = []
        coord_ts = []
        for ig, g in enumerate(self.grids):
            class_t = np.zeros((g[0], g[1], self.n_boxes[ig], self.n_classes + 1))
            anchor_t = self.generate_encoding_tensor_layer(self.img_size, self.grids[ig], self.anchor_dims[ig],
                                                           self.n_polygon)
            box_t = np.zeros((g[0], g[1], self.n_boxes[ig], self.n_polygon))
            box_t[:, :, :, :2] = 0.5
            box_t[:, :, :, 2:] = 1.0

            class_ts.append(class_t)
            anchor_ts.append(anchor_t)
            coord_ts.append(box_t)

        return class_ts, coord_ts, anchor_ts

    def _concatenate(self, class_ts, coord_ts):
        """
        Concatenate three parts to single tensor
        :param class_ts: (#output layers,#grid vertical,#grid horizontal,#n classes +1) class probabilites
        :param coord_ts: (#output layers,#grid vertical,#grid horizontal,#polygon) box dimensions
        :return: (#total boxes,[objectness,class probabilities one hot encoded, box cx, box cy, box width, box height])
                 Last elements contain decoding information such that a prediction can be transformed to a label in image size
                 without needing to know the configuration at training time.
        """
        y = []
        for ig, g in enumerate(self.grids):
            label_t = np.concatenate((coord_ts[ig], class_ts[ig]), -1)
            label_t[np.isnan(label_t[:, :, :, 0]), 0] = 0.0
            y.append(label_t)

        return y

    def _find_candidates(self, box_true):
        """
        Find anchors that can potentially be assigned responsible for prediction.
        :param box_true: True box
        :return: (#candidates,[idx_layer,idx_grid_x,idx_grid_y,idx_anchor,iou]) if candidates found None otherwise
        """
        candidates = []

        for ig, g in enumerate(self.grids):

            g_w = self.img_size[1] / g[1]
            g_h = self.img_size[0] / g[0]

            icx = int(np.floor(box_true.cx / g_w))
            icy = int(np.floor(box_true.cy / g_h))

            if icx >= g[1] or icx < 0 or 0 > icy or icy >= g[0]:
                break

            for ia in range(self.n_boxes[ig]):
                aw, ah = self.anchor_dims[ig][ia]
                acx = g_w * (icx + 0.5)
                acy = g_h * (icy + 0.5)
                anchor = Polygon.from_quad_t_centroid(np.array([[acx, acy, aw, ah]]))
                iou = box_true.iou(anchor)

                if iou > self.iou_min:
                    candidates.append(np.array([ig, icx, icy, ia, iou]))

        if not candidates:
            return None
        elif len(candidates) > 1:
            candidates = np.vstack(candidates)
        else:
            candidates = np.expand_dims(candidates[0], 0)
        candidates = candidates[candidates[:, -1].argsort()]
        return candidates

    def encode_label(self, label: ImgLabel):
        """

        Determine which anchor is responsible for prediction. Based on YoloV3 matching strategy.
        Thereby the anchor is assigned responsible that has the highest IoU with the GT box.
        Predictions of anchors that have an IoU > ignore_thresh are ignored in loss.
        GT labels that have IoU < iou_min are not assigned to any anchor.

        :param label: GT label containing all true bounding boxes in an image
        :return: (#total boxes,[objectness,class probabilities one hot encoded, box cx, box cy, box width, box height,
                                anchor with, anchor height, grid cx, grid cy, cell width, cell height])
                 Last elements contain decoding information such that a prediction can be transformed to a label in image size
                 without needing to know the configuration at training time.

        """
        class_ts, coord_ts, anchor_ts = self._create_empty_target_ts()

        for obj in label.objects:
            b = obj.poly.copy()
            b.points[:, 1] = self.img_size[0] - b.points[:, 1]

            candidates = self._find_candidates(b)

            if candidates is None:
                continue

            # ignore predictions of anchors that exceed iou_ignore
            for ic in range(candidates.shape[0]):
                if candidates[ic, -1] > self.iou_ignore:
                    ig, icx, icy, ia, _ = candidates.astype(np.int)[ic]
                    class_ts[ig][icy, icx, ia, 0] = -1.0

            # if best candidate exceeds iou_min assign target
            if candidates[0, -1] > self.iou_min:
                ig, icx, icy, ia, _ = candidates.astype(np.int)[0]
                class_one_hot = np.zeros((self.n_classes,)).T
                class_one_hot[obj.class_id] = 1.0
                objectness = 1.0
                x_off, y_off, p_w, p_h, cw, ch = anchor_ts[ig][icy, icx, ia]

                t_cx = (b.cx - x_off) / cw
                t_cy = (b.cy - y_off) / ch

                t_w = b.width / p_w
                t_h = b.height / p_h

                class_ts[ig][icy, icx, ia, 0] = objectness
                class_ts[ig][icy, icx, ia, 1:] = class_one_hot
                coord_ts[ig][icy, icx, ia] = t_cx, t_cy, t_w, t_h
                if self.verbose > 1:
                    print("Assigned Anchor: {}-{}-{}-{}: {}".format(ig, icx, icy, ia,
                                                                    coord_ts[ig][icy, icx, ia]))
        out = self._concatenate(class_ts, coord_ts)
        matched = np.sum([len(y[y[:, :, :, self.n_polygon] > 0]) for y in out])
        ignored = np.sum([len(y[y[:, :, :, self.n_polygon] < 0]) for y in out])
        self.unmatched += len(label.objects) - matched
        self.matched += matched
        self.ignored += ignored

        if any(np.any(np.isnan(y)) for y in out) or any(np.any(np.isinf(y)) for y in out):
            raise ValueError("Invalid Ground Truth")

        if any(np.any(y[:, :, :, :2] < 0) for y in out) or \
                any(np.any(y[:, :, :, :2] > 1) for y in out):
            raise ValueError('Invalid Ground Truth Center')

        if any(np.any(y[:, :, :, 2:4] <= 0) for y in out):
            raise ValueError('Invalid Width/Height')

        if self.verbose > 0:
            print("Assigned: {} Lost: {}. Ignored Anchors: {}".format(matched, len(label.objects) - matched, ignored))

        return out

    def encode_label_batch(self, labels: [ImgLabel]) -> np.array:
        ys = [] * len(self.grids)
        for i in range(len(self.grids)):
            ys.append([])
        for label in labels:
            y = self.encode_label(label)
            for i in range(len(self.grids)):
                ys[i].append(np.expand_dims(y[i], 0))
        for i in range(len(self.grids)):
            ys[i] = np.concatenate(ys[i], 0)
        return ys
