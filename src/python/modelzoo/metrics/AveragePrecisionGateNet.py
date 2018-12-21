import keras.backend as K

from modelzoo.metrics.MetricGateNet import MetricGateNet


class AveragePrecisionGateNet(MetricGateNet):
    def __init__(self, n_boxes, grid, iou_thresh, norm,batch_size, iou_thresh_nms=0.4,
                 confidence_levels=K.np.linspace(0, 1.0, 11)):
        super().__init__(n_boxes, grid, iou_thresh, norm, iou_thresh_nms, batch_size, confidence_levels)

    def compute(self, y_true, y_pred):
        """
        Calculates the average precision across all confidence levels

        :return: average precision
        """

        coord_true_t, class_true_t = self._postprocess_truth(y_true)

        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        average_precision = self.map_adapter.mean_average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t)

        return average_precision

    def total_precision(self, y_true, y_pred):
        """
        Calculates the average precision across all confidence levels

        :return: average precision
        """

        coord_true_t, class_true_t = self._postprocess_truth(y_true)

        coord_pred_t, class_pred_t = self._postprocess_pred(y_pred)

        total_precision = self.map_adapter.total_average_precision(coord_true_t, coord_pred_t, class_true_t, class_pred_t,self.confidence_levels)

        return total_precision