import keras.backend as K


class YoloV3Loss:
    def __init__(self, n_classes, n_polygon=4, weight_loc=5.0, weight_noobj=0.5,
                 weight_obj=5.0, weight_prob=1.0):
        self.n_classes = n_classes
        self.scale_prob = weight_prob
        self.scale_noob = weight_noobj
        self.scale_obj = weight_obj
        self.scale_coor = weight_loc
        self.n_polygon = n_polygon

    @staticmethod
    def _reshape(y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        n_boxes = K.shape(y_true)[1] * K.shape(y_true)[2] * K.shape(y_true)[3]
        y_true = K.reshape(y_true, (batch_size, n_boxes, -1))
        y_pred = K.reshape(y_pred, (batch_size, n_boxes, -1))
        return y_true, y_pred

    def total_loss(self, y_true, y_pred):
        """
        Loss function for YoloV3.
        :param y_true: y as fed for learning
        :param y_pred: raw network output
        :return: loss
        """
        loc_loss = self.localization_loss(y_true, y_pred)

        conf_loss = self.confidence_loss(y_true, y_pred)

        class_loss = self.classification_loss(y_true, y_pred)

        return loc_loss + conf_loss + class_loss

    def localization_loss(self, y_true, y_pred):
        y_true, y_pred = self._reshape(y_true, y_pred)
        positives = K.cast(K.equal(y_true[:, :, 0], 1), K.dtype(y_true))
        idx_coord = self.n_classes + 1
        coord_true = y_true[:, :, idx_coord:idx_coord + self.n_polygon]
        coord_pred = y_pred[:, :, idx_coord:idx_coord + self.n_polygon]

        xy_true = coord_true[:, :, :2]
        xy_true = K.clip(xy_true, K.epsilon(), 1 - K.epsilon())
        xy_true = K.log(xy_true / (1 - xy_true))

        xy_pred = coord_pred[:, :, :2]

        xy_loss = K.square(xy_true - xy_pred)

        xy_loss_sum = K.sum(xy_loss, -1) * self.scale_coor * positives

        wh_true = coord_true[:, :, 2:]
        wh_true = K.log(wh_true)
        wh_pred = coord_pred[:, :, 2:]
        wh_loss = K.square(wh_true - wh_pred)
        wh_loss_sum = K.sum(wh_loss, -1) * self.scale_coor * positives
        loc_loss = xy_loss_sum + wh_loss_sum
        total_loc_loss = K.sum(loc_loss) / K.cast(K.shape(y_true)[0], K.dtype(loc_loss))

        return total_loc_loss

    def confidence_loss(self, y_true, y_pred):
        y_true, y_pred = self._reshape(y_true, y_pred)
        positives = K.cast(K.equal(y_true[:, :, 0], 1), K.dtype(y_true))
        # ignore = K.cast(K.equal(y_true[:, :, 0], -1), K.dtype(y_true))
        negatives = K.cast(K.equal(y_true[:, :, 0], 0), K.dtype(y_true))

        weight = self.scale_noob * negatives + self.scale_obj * positives

        conf_pred = y_pred[:, :, 0:1]

        conf_loss = K.binary_crossentropy(target=K.expand_dims(positives, -1), output=conf_pred,
                                          from_logits=False) * K.expand_dims(weight, -1)
        total_conf_loss = K.sum(conf_loss) / K.cast(K.shape(y_true)[0], K.dtype(conf_loss))

        return total_conf_loss

    def classification_loss(self, y_true, y_pred):
        y_true, y_pred = self._reshape(y_true, y_pred)
        positives = K.cast(K.equal(y_true[:, :, 0], 1), K.dtype(y_true))
        negatives = K.cast(K.equal(y_true[:, :, 0], 0), K.dtype(y_true))

        weight = self.scale_noob * negatives + self.scale_obj * positives

        class_pred = y_pred[:, :, 1:self.n_classes]

        class_true = y_true[:, :, 1:self.n_classes]

        class_loss = K.categorical_crossentropy(target=class_true, output=class_pred,
                                                from_logits=True) * weight
        total_class_loss = K.sum(class_loss) / K.cast(K.shape(y_true)[0], K.dtype(class_loss))

        return total_class_loss
