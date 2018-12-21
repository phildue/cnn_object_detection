import keras.backend as K

from modelzoo.metrics.Metric import Metric


class AveragePrecision(Metric):
    def compute(self, y_true, y_pred):
        pass

    def __init__(self, iou_thresh, n_boxes, batch_size):
        self.batch_size = batch_size
        self.n_boxes = n_boxes
        self.iou_thresh = iou_thresh

    @staticmethod
    def class_match(class_a, class_b):
        """
        Implements the matching between two predictions as matrix operation.
        The tensors are reshaped to
        (len_a, #classes) -> (len_a, 1, 1) -> (len_a,len_b,1)
        (len_b, #classes) -> (1, len_b, 1) -> (len_a,len_b,1)
        Where the last column contains the predicted class
        :param class_a: tensor(len_a,#classes) containing predictions one-hot encoded
        :param class_b: tensor(len_b,#classes) containing predictions one-hot encoded
        :return: tensor(len_a,len_b) containing match between each box as float 1 for match
        """
        len_a = K.shape(class_a)[1]
        len_b = K.shape(class_b)[1]

        class_a_reshape = K.expand_dims(class_a, 2)
        class_b_reshape = K.expand_dims(class_b, 1)

        class_a_3d = K.tile(class_a_reshape, (1, 1, len_b, 1))
        class_b_3d = K.tile(class_b_reshape, (1, len_a, 1, 1))

        pred_a = K.argmax(class_a_3d, axis=-1)
        pred_b = K.argmax(class_b_3d, axis=-1)

        same_classes = K.cast(K.equal(pred_a, pred_b), K.floatx())

        return same_classes

    def _count_true_pos(self, match_iou):
        """
        Counts the number of true positives. We have to make sure each match gets only counted once.
        Hence we iterate over the rows and look for the max overlap. If it exceeds the iou threshold
        we count it as match. In the update step we zero the column corresponding to the box we
        considered as match to avoid double counting.
        :param match_iou: tensor(#true boxes,#pred boxes) where each element is the iou between box i and j
                            -if the corresponding box has a positive prediction (pred conf > conf_thresh)
                            -if the corresponding box had contains truly an obj (true conf > 0)
                            -if the both boxes contain the same class prediction
        :return: number of true positives
        """

        n_true_positives0 = K.zeros((self.batch_size,))
        w_valid0 = K.ones_like(match_iou)
        i0 = K.constant(0, dtype=K.tf.int32)

        def add_if_iou_match(idx, w_valid, n_true_positives):
            tp_iou_remaining = match_iou[:, idx, :] * w_valid[:, idx, :]
            max_iou = K.max(tp_iou_remaining, -1, keepdims=True)
            increment = K.cast(K.greater(max_iou, self.iou_thresh), K.floatx())
            n_true_positives = n_true_positives + K.reshape(increment, (self.batch_size,))

            # We build a matrix that contains 0 for the column that is considered as match 1 otherwise
            # Multiplying it with w_valid zeros the matched box and avoids double counting of the same box
            max_idx_mask = K.equal(max_iou, tp_iou_remaining)
            min_iou_mask = K.greater_equal(tp_iou_remaining, self.iou_thresh)

            w_update = K.switch(K.tf.logical_and(max_idx_mask, min_iou_mask),
                                K.zeros_like(tp_iou_remaining),
                                K.ones_like(tp_iou_remaining))

            w_update = K.tile(K.expand_dims(w_update, 1), (1, K.shape(match_iou)[1], 1))

            w_valid = w_update * w_valid
            return idx + 1, w_valid, n_true_positives

        def remaining_true_boxes(idx, w_valid, n_true_positives):
            return K.any(K.greater(K.sum(match_iou[:, idx, :] * w_valid[:, idx, :], -1), 0))

        i, w, tp_total = K.tf.while_loop(remaining_true_boxes, add_if_iou_match,
                                         [i0, w_valid0, n_true_positives0], parallel_iterations=1)
        return tp_total

    def _sort_by_conf(self, coord_t, class_t):
        """
        Sorts tensors by confidence level, starting with the highest confidence.
        :param coord_t: tensor(#boxes,4) bounding box coordinates
        :param class_t: tensor(#boxes,#classes) class confidences one-hot encoded
        :return:
        """
        confidence = K.max(class_t, -1)

        vals, idx_x = K.tf.nn.top_k(confidence, self.n_boxes, True)
        idx_x = K.expand_dims(idx_x, -1)

        idx_batch = K.arange(0, self.batch_size)
        idx_batch = K.expand_dims(idx_batch, -1)
        idx_batch = K.tile(idx_batch, (1, self.n_boxes))
        idx_batch = K.expand_dims(idx_batch, -1)
        idx = K.concatenate([idx_batch, idx_x], -1)
        class_sorted = K.tf.gather_nd(class_t, idx)
        coord_sorted = K.tf.gather_nd(coord_t, idx)
        return coord_sorted, class_sorted

    def _count_detections(self, class_pred, conf_thresh, iou, w_class_match, w_true):
        """
        Counts the number of true positives, false positives and false negatives for
        one particular confidence threshold.
        :param class_pred: tensor(#boxes,#classes) predicted class confidences one-hot encoded
        :param conf_thresh: minimum confidence to count as positive
        :param iou: tensor(#true boxes,#predicted boxes) containing the iou between each box
        :param w_class_match: tensor(#true boxes,#predicted boxes) containing 1 if two boxes have same class prediction 0 otherwise
        :param w_true: tensor(#true boxes,1) containing 1 if true box actually contains object
        :return: number of true positives, false positives and false negatives
        """
        w_positive = K.cast(K.greater(K.max(class_pred, -1, keepdims=True), conf_thresh), K.floatx())
        w_positive_transposed = K.permute_dimensions(w_positive, (0, 2, 1))

        n_true = K.sum(w_true, -1)

        w_true = K.expand_dims(w_true, 2)

        match_iou = iou * w_class_match * w_true * w_positive_transposed

        n_true_positives = self._count_true_pos(match_iou)

        w_positive = K.reshape(w_positive, (self.batch_size, -1))
        n_positives = K.sum(w_positive, -1)

        n_false_positives = n_positives - n_true_positives
        n_false_negatives = n_true - n_true_positives

        return n_true_positives, n_false_positives, n_false_negatives

    def prune_empty_boxes(self, coord_t, class_t):
        """
        Removes boxes that don't contain any predictions/true boxes.
        (1) Boxes are sorted by confidence
        (2) For each batch the index of the first smallest element is calculated
        (3) Boxes that are higher than the overall highest index are pruned. That way
            the dimension for each batch is the same and some of the boxes are still
            empty but the overall size of the tensor is drastically reduced.
        :param coord_t: tensor(#boxes,4) true bounding box coordinates in minmax-format
        :param class_t: tensor(#boxes,#classes) true class confidences one-hot encoded
        :return: same as input but with reduced size
        """
        coord_sorted, class_sorted = self._sort_by_conf(coord_t, class_t)

        conf_true = K.max(class_sorted, -1)

        min_idx = K.argmin(conf_true, axis=-1)
        max_min_idx = K.cast(K.max(min_idx), K.tf.int32)

        coord_sorted = coord_sorted[:, :max_min_idx + 1]
        class_sorted = class_sorted[:, :max_min_idx + 1]

        return coord_sorted, class_sorted

    def detections(self, coord_true, coord_pred, class_true, class_pred, conf_thresh=K.np.linspace(0.0, 1.0, 11)):
        """
        Determines number of true positives, false positives, false negatives.
        (1) boxes are sorted and boxes that don't contain positives are removed,
        (2) the iou between each box is calculated --> iou
        (3) the prediction between each box is compared --> w_class_match
        (4) Count detections counts true positives and so on
        (5) This is repeated for different conf levels

        :param coord_true: tensor(#boxes,4) true bounding box coordinates in minmax-format
        :param coord_pred: tensor(#boxes,4) predicted bounding box coordinates in minmax-format
        :param class_true: tensor(#boxes,#classes) true class confidences one-hot encoded
        :param class_pred: tensor(#boxes,#classes) predicted class confidences one-hot encoded
        :param conf_thresh: float only values above are considered as positive
        :return: true positives, false positives, false negatives
        """

        coord_true_sorted, class_true_sorted = self.prune_empty_boxes(coord_true, class_true)
        coord_pred_sorted, class_pred_sorted = self.prune_empty_boxes(coord_pred, class_pred)

        conf_true = K.max(class_true_sorted, -1)

        w_true = K.cast(K.greater(conf_true, 0), K.floatx())

        iou = Metric.iou(coord_true_sorted, coord_pred_sorted)

        w_class_match = self.class_match(class_true_sorted, class_pred_sorted)

        n_tp, n_fp, n_fn = [], [], []
        for i, c in enumerate(conf_thresh):
            n_tp_c, n_fp_c, n_fn_c = self._count_detections(class_pred_sorted, c,
                                                            iou,
                                                            w_class_match,
                                                            w_true)
            n_tp_c = K.expand_dims(n_tp_c, -1)
            n_fp_c = K.expand_dims(n_fp_c, -1)
            n_fn_c = K.expand_dims(n_fn_c, -1)
            n_tp.append(n_tp_c)
            n_fp.append(n_fp_c)
            n_fn.append(n_fn_c)

        n_tp = K.concatenate(n_tp, -1)
        n_fp = K.concatenate(n_fp, -1)
        n_fn = K.concatenate(n_fn, -1)

        return n_tp, n_fp, n_fn

    def precision_recall(self, coord_true, coord_pred, class_true, class_pred, conf_thresh=K.np.linspace(0.0, 1.0, 11)):
        """
        Calculates the precision-recall for one confidence level
        :param coord_true: tensor(#boxes,4) true bounding box coordinates in minmax-format
        :param coord_pred: tensor(#boxes,4) predicted bounding box coordinates in minmax-format
        :param class_true: tensor(#boxes,#classes) true class confidences one-hot encoded
        :param class_pred: tensor(#boxes,#classes) predicted class confidences one-hot encoded
        :param conf_thresh: float only values above are considered as positive
        :return: precision, recall
        """
        n_true_positives, n_false_positives, n_false_negatives = self.detections(coord_true, coord_pred, class_true,
                                                                                 class_pred, conf_thresh)

        precision = K.switch(K.greater(n_true_positives, 0),
                             n_true_positives / (n_false_positives + n_true_positives),
                             K.zeros_like(n_true_positives))

        recall = K.switch(K.greater(n_true_positives, 0),
                          n_true_positives / (n_true_positives + n_false_negatives),
                          K.zeros_like(n_true_positives))
        total_predictions = n_true_positives + n_false_positives

        return precision, recall, total_predictions

    def mean_average_precision(self, coord_true, coord_pred, class_true, class_pred):
        """
        Calculates the average precision across all confidence levels
        :param coord_true: tensor(#boxes,4) true bounding box coordinates in minmax-format
        :param coord_pred: tensor(#boxes,4) predicted bounding box coordinates in minmax-format
        :param class_true: tensor(#boxes,#classes) true class confidences one-hot encoded
        :param class_pred: tensor(#boxes,#classes) predicted class confidences one-hot encoded
        :return: average precision
        """
        confidence_levels = K.np.linspace(0, 1.0, 11)

        precision, recall, _ = self.precision_recall(coord_true, coord_pred, class_true, class_pred,
                                                     confidence_levels)
        average_precision = K.mean(precision, -1)

        return average_precision

    def total_average_precision(self, coord_true, coord_pred, class_true, class_pred,
                                confidence_levels=K.np.linspace(0, 1.0, 11)):
        n_true_positives, n_false_positives, n_false_negatives = self.detections(coord_true, coord_pred, class_true,
                                                                                 class_pred, confidence_levels)

        total_tp = K.sum(n_true_positives, -1)
        total_fp = K.sum(n_false_positives, -1)
        total_fn = K.sum(n_false_negatives, -1)
        precision = K.switch(K.greater(total_tp, 0),
                             total_fn / (total_fp + total_tp),
                             K.zeros_like(total_tp))

        average_precision = K.mean(precision, -1)

        recall = K.switch(K.greater(total_tp, 0),
                          total_tp / (total_tp + total_fn),
                          K.zeros_like(total_tp))

        average_recall = K.mean(recall, -1)

        return average_precision, average_recall
