import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Concatenate
from modelzoo.layers.DepthwiseConv2D import DepthwiseConv2D


def darknet_conv(layer_in, filters, size, stride):
    conv = Conv2D(filters, kernel_size=(size, size), strides=(stride, stride), padding='same', use_bias=False)(layer_in)
    norm = BatchNormalization()(conv)
    act = LeakyReLU(0.1)(norm)
    return act


def dconv(netin, filters, kernel_size, strides, alpha):
    conv1 = DepthwiseConv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)
    conv2 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)
    return act2


def bottleneck_dconv(netin, filters, kernel_size, strides, expansion, alpha):
    expand = Conv2D(int(K.int_shape(netin)[-1] * expansion), (1, 1), strides=(1, 1), padding='same', use_bias=False)(
        netin)
    norm1 = BatchNormalization()(expand)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    dconv = DepthwiseConv2D(int(K.int_shape(netin)[-1] * expansion), kernel_size=kernel_size, strides=strides,
                            padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(dconv)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    compress = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(
        act2)
    norm3 = BatchNormalization()(compress)

    return norm3


def bottleneck_dconv_residual(netin, filters, kernel_size, strides, compression, alpha):
    fork = bottleneck_dconv(netin, filters, kernel_size, strides, compression, alpha)
    join = Add()([netin, fork])
    return join


def bottleneck_conv(netin, filters, kernel_size, strides, compression, alpha):
    compress = Conv2D(int(K.int_shape(netin)[-1] * compression), (1, 1), strides=(1, 1), padding='same',
                      use_bias=False)(
        netin)
    norm1 = BatchNormalization()(compress)

    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                  padding='same', use_bias=False)(norm1)
    norm2 = BatchNormalization()(conv)
    act = LeakyReLU(alpha=alpha)(norm2)

    return act


def bottleneck_conv_residual(netin, filters, kernel_size, strides, compression, alpha):
    fork = bottleneck_conv(netin, filters, kernel_size, strides, compression, alpha)
    join = Add()([netin, fork])
    return join


def wr_basic_conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv1 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    conv2 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    return Add()([netin, act2])


def wr_bottleneck_conv_leaky(netin, filters, compression_factor, kernel_size, strides, alpha):
    conv1 = Conv2D(int((K.int_shape(netin))[-1] * compression_factor), kernel_size=(1, 1), strides=strides,
                   padding='same', use_bias=False)(
        netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    conv2 = Conv2D(int(filters), kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    join = Add()([netin, act2])
    return join


def wr_inception_conv_leaky(netin, filters, compression, kernel_size, strides, alpha):
    conv_squeeze = Conv2D(int((K.int_shape(netin))[-1] * compression), kernel_size=(1, 1), strides=strides,
                          padding='same', use_bias=False)(netin)
    norm_squeeze = BatchNormalization()(conv_squeeze)
    act_squeeze = LeakyReLU(alpha=alpha)(norm_squeeze)

    conv11 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        act_squeeze)
    norm11 = BatchNormalization()(conv11)
    act11 = LeakyReLU(alpha=alpha)(norm11)

    conv12 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act11)
    norm12 = BatchNormalization()(conv12)
    act12 = LeakyReLU(alpha=alpha)(norm12)

    conv21 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        act_squeeze)
    norm21 = BatchNormalization()(conv21)
    act21 = LeakyReLU(alpha=alpha)(norm21)

    concat = Concatenate()([act12, act21])
    join = Add()([netin, concat])
    return join


def conv_concat(netin, filters, compression, kernel_size, strides, alpha):
    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                  padding='same', use_bias=False)(netin)
    norm = BatchNormalization()(conv)
    act = LeakyReLU(alpha=alpha)(norm)
    concat = Concatenate()([act, netin])
    if compression != 1.0:
        out = Conv2D(int((K.int_shape(concat))[-1] * compression), kernel_size=(1, 1), strides=strides,
                     padding='same', use_bias=False)(concat)
    else:
        out = concat

    return out
