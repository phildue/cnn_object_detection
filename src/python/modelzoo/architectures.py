from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, UpSampling2D, Concatenate, Cropping2D, \
    Reshape
from modelzoo.YoloV3Encoder import YoloV3Encoder
from modelzoo.layers.ConcatMeta import ConcatMeta
import keras.backend as K
import numpy as np


def darknet_conv(layer_in, filters, size, stride):
    conv = Conv2D(filters, kernel_size=(size, size), strides=(stride, stride), padding='same', use_bias=False)(layer_in)
    norm = BatchNormalization()(conv)
    act = LeakyReLU(0.1)(norm)
    return act


def build_yolov3_tiny(n_classes):
    anchors = np.array([
        [[81, 82],
         [135, 169],
         [344, 319]],
        [[10, 14],
         [23, 27],
         [37, 58]],
    ])
    grids = [
        [13, 13],
        [26, 26]
    ]
    img_size = (416, 416)
    n_boxes = [len(a) for a in anchors]

    netin = Input((img_size[0], img_size[1], 3))
    conv1 = darknet_conv(netin, 16, 3, 1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = darknet_conv(pool1, 32, 3, 1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = darknet_conv(pool2, 64, 3, 1)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = darknet_conv(pool3, 128, 3, 1)
    pool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = darknet_conv(pool4, 256, 3, 1)
    pool5 = MaxPooling2D((2, 2))(conv5)
    conv6 = darknet_conv(pool5, 512, 3, 1)
    pool7 = MaxPooling2D((2, 2), (1, 1))(conv6)
    conv7 = darknet_conv(pool7, 1024, 3, 1)
    conv8 = darknet_conv(conv7, 256, 1, 1)
    conv9 = darknet_conv(conv8, 512, 3, 1)

    pred1 = Conv2D(n_boxes[0] * (n_classes + 4 + 1), kernel_size=(1, 1), strides=(1, 1))(conv9)

    conv91 = darknet_conv(conv9, 128, 1, 1)
    upsample91 = UpSampling2D()(conv91)
    route = Cropping2D(((1, 1), (1, 1)))(conv5)
    concat81 = Concatenate()([upsample91, route])
    conv10 = darknet_conv(concat81, 256, 3, 1)
    pred2 = Conv2D(n_boxes[1] * (n_classes + 4 + 1), kernel_size=(1, 1), strides=(1, 1))(conv10)

    pred1 = Reshape((-1, (n_classes + 4 + 1)))(pred1)
    pred2 = Reshape((-1, (n_classes + 4 + 1)))(pred2)
    pred = Concatenate(1)([pred1, pred2])

    meta_t = K.constant(YoloV3Encoder.generate_encoding_tensor(img_size, grids, anchors, 4),
                        K.tf.float32)

    netout = ConcatMeta(meta_t)(pred)

    model = Model(netin, netout)

    return model, grids, anchors, img_size
