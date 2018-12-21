from abc import abstractmethod

import keras.backend as K
import numpy as np

from modelzoo.metrics.AveragePrecision import AveragePrecision
from modelzoo.metrics.Metric import Metric


class MetricGateNet(Metric):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, n_boxes=[5], grid=[(13, 13)], iou_thresh=0.4, norm=(416, 416), iou_thresh_nms=0.4,
                 batch_size=8, confidence_levels=K.np.linspace(0, 1.0, 11)):
        self.confidence_levels = confidence_levels
        self.batch_size = batch_size
        self.iou_thresh = iou_thresh_nms
        self.norm = norm
        self.n_boxes = n_boxes
        self.grid = grid
        total_boxes = np.sum([grid[i][0] * grid[i][1] * n_boxes[i] for i in range(len(n_boxes))])
        self.map_adapter = AveragePrecision(iou_thresh, total_boxes, batch_size=batch_size)

    def _decode_coord(self, coord_t, anchors_t):
        t_cx = coord_t[:, :, 0]
        t_cy = coord_t[:, :, 1]
        t_w = coord_t[:, :, 2]
        t_h = coord_t[:, :, 3]

        xoff = anchors_t[:, :, -4]
        yoff = anchors_t[:, :, -3]
        cw = anchors_t[:, :, -2]
        ch = anchors_t[:, :, -1]
        p_w = anchors_t[:, :, -2]
        p_h = anchors_t[:, :, -1]

        b_cx = K.sigmoid(t_cx)*cw+xoff
        b_cy = K.sigmoid(t_cy)*ch+yoff
        b_w = K.exp(t_w)*p_w
        b_h = K.exp(t_h)*p_h

        b_cy = self.norm[0] - b_cy

        b_xmin = b_cx - b_w / 2
        b_ymin = b_cy - b_h / 2
        b_xmax = b_cx + b_w / 2
        b_max = b_cy + b_h / 2

        b_xmin = K.expand_dims(b_xmin, -1)
        b_ymin = K.expand_dims(b_ymin, -1)
        b_xmax = K.expand_dims(b_xmax, -1)
        b_max = K.expand_dims(b_max, -1)
        coord_dec_t = K.concatenate([b_xmin, b_ymin, b_xmax, b_max], -1)

        return coord_dec_t

    def _postprocess_truth(self, y_true):
        coord_true_t = y_true[:, :, 1:5]
        conf_true_t = y_true[:, :, :1]
        anchors_t = y_true[:, :, 5:]

        coord_true_dec_t = self._decode_coord(coord_true_t, anchors_t)

        return coord_true_dec_t, conf_true_t

    def _postprocess_pred(self, y_pred):
        xy_pred_t = K.sigmoid(y_pred[:, :, 1:3])
        wh_pred_t = K.exp(y_pred[:, :, 3:5])
        # wh_pred_t = y_pred[:, :, 3:5]
        coord_pred_t = K.concatenate((xy_pred_t, wh_pred_t), -1)
        # coord_pred_t = y_pred[:, :, 1:5]
        conf_pred_t = y_pred[:, :, :1]
        anchors_t = y_pred[:, :, 5:]

        coord_pred_dec_t = self._decode_coord(coord_pred_t, anchors_t)

        class_pred_nms_batch = self.map_adapter.non_max_suppression_batch(coord_pred_dec_t,
                                                                          conf_pred_t,
                                                                          self.batch_size,
                                                                          K.shape(anchors_t)[0],
                                                                          self.iou_thresh)

        return coord_pred_dec_t, class_pred_nms_batch
