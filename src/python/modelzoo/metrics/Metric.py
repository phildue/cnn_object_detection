from abc import ABC, abstractmethod

import keras.backend as K


class Metric(ABC):
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    @staticmethod
    def iou(box_a, box_b):
        """
        Implements the calculation of the intersection-over-union between each box in a and b as matrix operation.
        The tensors are reshaped to
        (#batch,len_a, 2) -> (#batch,len_a, 1, 2) -> (#batch,len_a,len_b,2)
        (#batch,len_b, 2) -> (#batch,1, len_b, 2) -> (#batch,len_a,len_b,2)
        Then the corners can be calculated between each box by matrix operations.
        :param box_a: tensor(#batch,len_a,4) containing box coordinates in min-max format
        :param box_b: tensor(#batch,len_b,4) containing box coordinates in min-max format
        :return: tensor(#batch,len_a,len_b) containing iou between each box
        """
        len_a = K.shape(box_a)[1]
        len_b = K.shape(box_b)[1]

        box_a_expand = K.expand_dims(box_a[:, :, 2:], 2)
        box_b_expand = K.expand_dims(box_b[:, :, 2:], 1)

        box_a_reshape3d = K.tile(box_a_expand, [1, 1, len_b, 1])
        box_b_reshape3d = K.tile(box_b_expand, [1, len_a, 1, 1])
        max_xy = K.min([box_a_reshape3d, box_b_reshape3d], axis=0)

        box_a_expand = K.expand_dims(box_a[:, :, :2], 2)
        box_b_expand = K.expand_dims(box_b[:, :, :2], 1)

        box_a_reshape3d = K.tile(box_a_expand, [1, 1, len_b, 1])
        box_b_reshape3d = K.tile(box_b_expand, [1, len_a, 1, 1])
        min_xy = K.max([box_a_reshape3d, box_b_reshape3d], axis=0)

        width_height = (max_xy - min_xy)
        width_height = K.maximum(width_height, 0)

        area_intersect = width_height[:, :, :, 0] * width_height[:, :, :, 1]

        area_a = K.tile(
            K.reshape((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), (-1, len_a, 1)),
            (1, 1, len_b))
        area_b = K.tile(
            K.reshape((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), (-1, 1, len_b)),
            (1, len_a, 1))

        union = area_a + area_b - area_intersect

        iou = area_intersect / union

        iou = K.cast(iou, K.floatx())

        return iou

    @staticmethod
    def non_max_suppression_batch(coord_pred_t, class_pred_t, batch_size, n_boxes_max, iou_thresh):
        """
        Performs non-max suppression on a batch of labels.
        :param coord_pred_t: tensor(#batch,#boxes,4) containing box coordinates in min-max format
        :param class_pred_t: tensor(#batch,#boxes,#classes) containing predictions per box
        :param batch_size: size of the batch
        :param n_boxes_max: max number of boxes to keep
        :param iou_thresh: intersection-over-union threshold when a box counts as overlapping
        :return: tensor(#batch,#boxes,#classes) where the suppressed boxes are all 0s
        """
        conf_pred_t = K.max(class_pred_t, axis=-1)

        class_pred_nms_batch = []
        for i in range(batch_size):
            idx = K.tf.image.non_max_suppression(K.cast(coord_pred_t[i], K.tf.float32),
                                                 K.cast(K.flatten(conf_pred_t[i]), K.tf.float32),
                                                 n_boxes_max, iou_thresh,
                                                 'NonMaxSuppression')
            idx = K.expand_dims(idx, 1)
            class_pred_nms = K.tf.gather_nd(class_pred_t[i], idx)
            class_pred_nms = K.tf.scatter_nd(idx, class_pred_nms, shape=K.shape(class_pred_t[0]))
            class_pred_nms = K.expand_dims(class_pred_nms, 0)
            class_pred_nms_batch.append(class_pred_nms)

        class_pred_nms_batch = K.concatenate(class_pred_nms_batch, 0)
        return class_pred_nms_batch
