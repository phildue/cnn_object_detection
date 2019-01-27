import keras.backend as K
import numpy as np

from utils.labels.ObjectLabel import ObjectLabel


def iou_np(box_a, box_b):
    """
    Implements the calculation of the intersection-over-union between each box in a and b as matrix operation.
    The tensors are reshaped to
    (len_a, 2) -> (len_a, 1, 2) -> (len_a,len_b,2)
    (len_b, 2) -> (1, len_b, 2) -> (len_a,len_b,2)
    Then the corners can be calculated between each box by matrix operations.
    :param box_a: tensor(len_a,4) containing box coordinates in min-max format
    :param box_b: tensor(len_b,4) containing box coordinates in min-max format
    :return: tensor(len_a,len_b) containing iou between each box
    """

    len_a = box_a.shape[0]
    len_b = box_b.shape[0]

    box_a_reshape = np.reshape(box_a[:, 2:], (len_a, 1, 2))
    box_b_reshape = np.reshape(box_b[:, 2:], (1, len_b, 2))

    box_a_reshape3d = np.tile(box_a_reshape, (1, len_b, 1))
    box_b_reshape3d = np.tile(box_b_reshape, (len_a, 1, 1))
    max_xy = np.min([box_a_reshape3d, box_b_reshape3d], axis=0)

    box_a_reshape = np.reshape(box_a[:, :2], (len_a, 1, 2))
    box_b_reshape = np.reshape(box_b[:, :2], (1, len_b, 2))

    box_a_reshape3d = np.tile(box_a_reshape, (1, len_b, 1))
    box_b_reshape3d = np.tile(box_b_reshape, (len_a, 1, 1))
    min_xy = np.max([box_a_reshape3d, box_b_reshape3d], axis=0)

    width_height = (max_xy - min_xy)
    width_height[width_height < 0] = 0
    area_intersect = width_height[:, :, 0] * width_height[:, :, 1]

    area_a = np.tile(np.reshape((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), (len_a, 1)),
                     (1, len_b))
    area_b = np.tile(np.reshape((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), (1, len_b)),
                     (len_a, 1))

    union = area_a + area_b - area_intersect

    return area_intersect / union


def non_max_suppression(boxes: [ObjectLabel], iou_thresh=0.4, n_max=50):
    coord_t = np.concatenate([np.expand_dims(b.poly.to_quad_t_minmax, 0) for b in boxes], 0)
    confs = [b.confidence for b in boxes]
    conf_t = np.array(confs).flatten()

    idx = K.get_session().run(non_max_suppression_tf(K.constant(coord_t), K.constant(conf_t), iou_thresh, n_max))

    return [b for i, b in enumerate(boxes) if i in idx]


def non_max_suppression_tf(boxes_pred_t, class_conf_t, iou_thresh, n_max=50):
    return K.tf.image.non_max_suppression(boxes_pred_t, class_conf_t, n_max, iou_thresh, name='NonMaxSuppression')
