import keras.backend as K

from modelzoo.metrics.Loss import Loss


class YoloV3Loss(Loss):
    def __init__(self, n_polygon=4, weight_loc=5.0, weight_noobj=0.5,
                 weight_obj=5.0, weight_prob=1.0):
        self.scale_prob = weight_prob
        self.scale_noob = weight_noobj
        self.scale_obj = weight_obj
        self.scale_coor = weight_loc
        self.n_polygon = n_polygon

    def loss(self, y_true, y_pred):
        y_true_k = K.constant(y_true, name="y_true")
        y_pred_k = K.constant(y_pred, name="netout")

        return K.get_session().run(self.compute(y_true=y_true_k, y_pred=y_pred_k))

    def compute(self, y_true, y_pred):
        """
        Loss function for GateNet.
        :param y_true: y as fed for learning
        :param y_pred: raw network output
        :return: loss
        """
        # y_pred = K.print_tensor(y_pred, "Y_Pred")
        # y_true = K.print_tensor(y_true, "Y_True")
        loc_loss = self.localization_loss(y_true, y_pred)

        conf_loss = self.confidence_loss(y_true, y_pred)

        return loc_loss + conf_loss

    def localization_loss(self, y_true, y_pred):
        positives = K.cast(K.equal(y_true[:, :, 0], 1), K.dtype(y_true))
        coord_true = y_true[:, :, 1:5]
        coord_pred = y_pred[:, :, 1:5]

        xy_true = coord_true[:, :, :2]
        # xy_true = K.clip(xy_true, K.epsilon(), 1 - K.epsilon())
        # xy_true = K.log(xy_true / (1 - xy_true))
        # xy_pred = K.sigmoid(coord_pred[:, :, :2])
        xy_pred = coord_pred[:, :, :2]
        # xy_loss = 0.5*K.square(xy_true - xy_pred)
        xy_loss = K.binary_crossentropy(target=xy_true, output=xy_pred, from_logits=True)
        xy_loss_sum = K.sum(xy_loss, -1) * self.scale_coor * positives

        wh_true = K.log(coord_true[:, :, 2:])
        wh_pred = coord_pred[:, :, 2:]
        wh_loss = K.square(wh_true - wh_pred)
        wh_loss_sum = K.sum(wh_loss, -1) * self.scale_coor * positives
        loc_loss = xy_loss_sum + wh_loss_sum
        total_loc_loss = K.sum(loc_loss) / K.cast(K.shape(y_true)[0], K.dtype(loc_loss))

        # loc_loss_sum = K.print_tensor(loc_loss_sum,'Loc Loss=')
        # total_loc_loss = K.tf.Print(total_loc_loss, [K.sum(wh_loss_sum, -1), K.sum(xy_loss_sum
        #                                                                            , -1)], 'Localization Loss=')
        return total_loc_loss

    def confidence_loss(self, y_true, y_pred):
        positives = K.cast(K.equal(y_true[:, :, 0], 1), K.dtype(y_true))
        # ignore = K.cast(K.equal(y_true[:, :, 0], -1), K.dtype(y_true))
        negatives = K.cast(K.equal(y_true[:, :, 0], 0), K.dtype(y_true))

        weight = self.scale_noob * negatives + self.scale_obj * positives

        conf_pred = y_pred[:, :, 0:1]

        conf_loss = K.binary_crossentropy(target=K.expand_dims(positives, -1), output=conf_pred,
                                          from_logits=True) * K.expand_dims(weight, -1)
        total_conf_loss = K.sum(conf_loss) / K.cast(K.shape(y_true)[0], K.dtype(conf_loss))

        # conf_loss_total = K.print_tensor(conf_loss_total,'Conf Loss=')

        return total_conf_loss
