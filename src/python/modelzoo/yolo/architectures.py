import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Reshape
from modelzoo.layers.layers import darknet_conv
import keras.backend as K


def build_yolov3_tiny(n_classes):
    anchors = np.array(
        [
            [[81, 82],
             [135, 169],
             [344, 319]],
            [[10, 14],
             [23, 27],
             [37, 58]],
        ])

    n_boxes = [len(a) for a in anchors]

    # Base
    netin = Input((None, None, 3))
    conv1 = darknet_conv(netin, 16, 3, 1)
    pool1 = MaxPooling2D((2, 2))(conv1)  # 208
    conv2 = darknet_conv(pool1, 32, 3, 1)
    pool2 = MaxPooling2D((2, 2))(conv2)  # 104
    conv3 = darknet_conv(pool2, 64, 3, 1)
    pool3 = MaxPooling2D((2, 2))(conv3)  # 52
    conv4 = darknet_conv(pool3, 128, 3, 1)
    pool4 = MaxPooling2D((2, 2))(conv4)  # 26
    conv5 = darknet_conv(pool4, 256, 3, 1)
    pool5 = MaxPooling2D((2, 2))(conv5)  # 13
    conv6 = darknet_conv(pool5, 512, 3, 1)
    pool7 = MaxPooling2D((2, 2), (1, 1), padding='same')(conv6)
    conv7 = darknet_conv(pool7, 1024, 3, 1)
    conv8 = darknet_conv(conv7, 256, 1, 1)
    conv9 = darknet_conv(conv8, 512, 3, 1)

    # First Output Branch
    pred1 = Conv2D(n_boxes[0] * (n_classes + 4 + 1), kernel_size=(1, 1), strides=(1, 1))(conv9)
    # Second Output Branch
    conv91 = darknet_conv(conv8, 128, 1, 1)
    upsample91 = UpSampling2D()(conv91)
    concat81 = Concatenate()([upsample91, conv5])
    conv10 = darknet_conv(concat81, 256, 3, 1)
    pred2 = Conv2D(n_boxes[1] * (n_classes + 4 + 1), kernel_size=(1, 1), strides=(1, 1))(conv10)

    netout = [pred1, pred2]
    model = Model(netin, netout)

    return model, anchors


def build_yolov3_test(n_classes):
    anchors = np.array(
        [
            [[81, 82],
             [135, 169],
             [344, 319]],
            [[10, 14],
             [23, 27],
             [37, 58]],
        ])

    n_boxes = [len(a) for a in anchors]

    # Base
    netin = Input((None, None, 3))
    conv1 = darknet_conv(netin, 2, 3, 1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)  # 208
    conv2 = darknet_conv(pool1, 4, 3, 1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)  # 104
    conv3 = darknet_conv(pool2, 6, 3, 1)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv3)  # 52
    conv4 = darknet_conv(pool3, 8, 3, 1)
    pool4 = MaxPooling2D((2, 2), padding='same')(conv4)  # 26
    conv5 = darknet_conv(pool4, 10, 3, 1)
    pool5 = MaxPooling2D((2, 2), padding='same')(conv5)  # 13
    conv6 = darknet_conv(pool5, 12, 3, 1)
    pool7 = MaxPooling2D((2, 2), (1, 1), padding='same')(conv6)
    conv7 = darknet_conv(pool7, 14, 3, 1)
    conv8 = darknet_conv(conv7, 16, 1, 1)
    conv9 = darknet_conv(conv8, 18, 3, 1)

    # First Output Branch
    pred1 = Conv2D(n_boxes[0] * (n_classes + 4 + 1), kernel_size=(1, 1), strides=(1, 1), name='LargeOut')(conv9)

    # Second Output Branch
    conv91 = darknet_conv(conv9, 128, 1, 1)
    upsample91 = UpSampling2D()(conv91)
    # route = Cropping2D(((1, 1), (1, 1)))(conv5)
    concat81 = Concatenate()([upsample91, conv5])
    conv10 = darknet_conv(concat81, 256, 3, 1)
    pred2 = Conv2D(n_boxes[1] * (n_classes + 4 + 1), kernel_size=(1, 1), strides=(1, 1), name='SmallOut')(conv10)

    netout = [pred1, pred2]
    model = Model(netin, netout)

    return model, anchors
