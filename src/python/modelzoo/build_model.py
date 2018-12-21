import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, TimeDistributed, Add, Concatenate, \
    UpSampling2D, AveragePooling2D, Cropping2D
from keras.layers import Reshape
from sklearn.cluster import KMeans

from modelzoo.Decoder import Decoder
from modelzoo.Detector import Detector
from modelzoo.Encoder import Encoder
from modelzoo.Postprocessor import Postprocessor
from modelzoo.Preprocessor import Preprocessor
from modelzoo.layers.ConcatMeta import ConcatMeta
from modelzoo.layers.DepthwiseConv2D import DepthwiseConv2D
from utils.ModelSummary import ModelSummary
from utils.fileaccess.labelparser.DatasetParser import DatasetParser


def load_detector(directory, img_shape=None, preprocessing=None):
    summary = ModelSummary.from_file(directory + '/summary.pkl')
    img_res = summary.img_res if img_shape is None else img_shape
    anchors = summary.anchors
    architecture = summary.architecture
    color_format = summary.color_format
    model, output_grids = build_detector(img_shape=(img_res[0], img_res[1], 3), architecture=architecture,
                                         anchors=anchors,
                                         n_polygon=4)
    model.load_weights(directory + '/model.h5')
    encoder = Encoder(anchor_dims=anchors, img_norm=img_res, grids=output_grids, n_polygon=4, iou_min=0.4)
    preprocessor = Preprocessor(preprocessing=preprocessing, encoder=encoder, n_classes=1, img_shape=img_res,
                                color_format=color_format)
    decoder = Decoder(anchor_dims=anchors, n_polygon=4, norm=img_res, grid=output_grids)
    postproessor = Postprocessor(decoder=decoder)

    detector = Detector(model, preprocessor, postproessor, summary)

    return detector


def kmeans_anchors(n_boxes: [int], label_source, img_shape):
    if isinstance(label_source, list):
        labels = []
        for p in label_source:
            label_reader = DatasetParser.get_parser(p, 'xml', color_format='bgr')
            labels.extend(label_reader.read()[1])
    else:
        label_reader = DatasetParser.get_parser(label_source, 'xml', color_format='bgr')
        _, labels = label_reader.read()

    wh = []
    for label in labels:
        h, w = img_shape
        for b in label.objects:
            if 0 < b.poly.width < w \
                    and 0 < b.poly.height < h:
                box_dim = np.array([b.poly.width, b.poly.height])
                box_dim = np.expand_dims(box_dim, 0)
                wh.append(box_dim)
    box_dims = np.concatenate(wh, 0)

    kmeans = KMeans(n_clusters=np.sum(n_boxes)).fit(box_dims)
    centers = kmeans.cluster_centers_
    centers = np.round(centers, 2)
    centers = np.sort(centers,0)[::-1]
    anchors = []
    idx = 0
    for n in n_boxes:
        anchors.append(centers[idx:idx + n])
        idx += n

    return np.array(anchors)


def build_detector(img_shape, architecture, anchors, n_polygon=4):
    h, w, input_channels = img_shape
    n_boxes = [len(a) for a in anchors]

    netin = Input((h, w, input_channels))

    net = netin
    grids = []
    prediction_layer_i = 0
    predictions = []
    layers = []
    for i, config in enumerate(architecture):
        if 'predict' in config['name']:
            with K.name_scope('predict{}'.format(prediction_layer_i)):
                inference = Conv2D(n_boxes[prediction_layer_i] * (n_polygon + 1), kernel_size=(1, 1),
                                   strides=(1, 1), activation='linear', name='predictor{}'.format(prediction_layer_i))(
                    net)
                prediction_layer_i += 1
                reshape = Reshape((-1, n_polygon + 1))(inference)
                predictions.append(reshape)
                layers.append(inference)
                grid = K.int_shape(net)[-3], K.int_shape(net)[-2]
                grids.append(grid)
        elif 'route' in config['name']:
            if len(config['index']) > 1:
                net = Concatenate()([layers[i] for i in config['index']])
            else:
                net = layers[config['index'][0]]
            layers.append(net)
        else:
            with K.name_scope('layer' + str(i)):
                net = create_layer(net, config)
                layers.append(net)
    if len(predictions) > 1:
        predictions = Concatenate(-2)(predictions)
    else:
        predictions = predictions[0]

    meta_t = K.constant(Encoder.generate_encoding((h, w), grids, anchors, n_polygon),
                        K.tf.float32)

    netout = ConcatMeta(meta_t)(predictions)
    model = Model(netin, netout)
    return model, grids


def create_layer(netin, config):
    return layers[config['name']](netin, config)


def conv_leaky_creator(netin, config):
    return conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm = BatchNormalization()(conv)
    act = LeakyReLU(alpha=alpha)(norm)
    return act


def avg_pool_creator(netin, config):
    return avg_pool(netin, config['size'], config['strides'])


def avg_pool(netin, size, strides):
    return AveragePooling2D(size, strides, padding='same')(netin)


def max_pool_creator(netin, config):
    try:
        strides = config['strides']
    except KeyError:
        strides = config['size']
    return max_pool(netin, config['size'], strides)


def max_pool(netin, size, strides):
    return MaxPooling2D(size, strides, padding='same')(netin)


def time_dist_conv_leaky_creator(netin, config):
    return time_dist_conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def time_dist_conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv = TimeDistributed(Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False))(
        netin)
    norm = TimeDistributed(BatchNormalization())(conv)
    act = TimeDistributed(LeakyReLU(alpha=alpha))(norm)
    return act


def time_dist_max_pool_creator(netin, config):
    return time_dist_max_pool(netin, config['size'])


def time_dist_max_pool(netin, size):
    return TimeDistributed(MaxPooling2D(size))(netin)


def dconv_creator(netin, config):
    return conv_leaky(netin, config['filters'], config['kernel_size'], config['strides'], config['alpha'])


def dconv(netin, filters, kernel_size, strides, alpha):
    conv1 = DepthwiseConv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)
    conv2 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)
    return act2


def bottleneck_dconv_creator(netin, config):
    return bottleneck_dconv(netin, config['filters'], config['kernel_size'], config['strides'], config['expansion'],
                            config['alpha'])


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


def bottleneck_dconv_residual_creator(netin, config):
    return bottleneck_dconv_residual(netin, config['filters'], config['kernel_size'], config['strides'],
                                     config['expansion'], config['alpha'])


def bottleneck_dconv_residual(netin, filters, kernel_size, strides, compression, alpha):
    fork = bottleneck_dconv(netin, filters, kernel_size, strides, compression, alpha)
    join = Add()([netin, fork])
    return join


def bottleneck_conv_creator(netin, config):
    return bottleneck_conv(netin, config['filters'], config['kernel_size'], config['strides'], config['compression'],
                           config['alpha'])


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


def bottleneck_conv_residual_creator(netin, config):
    return bottleneck_conv_residual(netin, config['filters'], config['kernel_size'], config['strides'],
                                    config['compression'], config['alpha'])


def bottleneck_conv_residual(netin, filters, kernel_size, strides, compression, alpha):
    fork = bottleneck_conv(netin, filters, kernel_size, strides, compression, alpha)
    join = Add()([netin, fork])
    return join


def wr_basic_conv_leaky_creator(netin, config):
    return wr_basic_conv_leaky(netin, config['filters'], config['kernel_size'],
                               config['strides'], config['alpha'])


def wr_basic_conv_leaky(netin, filters, kernel_size, strides, alpha):
    conv1 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(netin)
    norm1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=alpha)(norm1)

    conv2 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=alpha)(norm2)

    return Add()([netin, act2])


def wr_bottleneck_conv_leaky_creator(netin, config):
    return wr_bottleneck_conv_leaky(netin, config['filters'], config['compression'], config['kernel_size'],
                                    config['strides'],
                                    config['alpha'])


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


def wr_inception_conv_leaky_creator(netin, config):
    return wr_inception_conv_leaky(netin, config['filters'], config['compression'], config['kernel_size'],
                                   config['strides'],
                                   config['alpha'])


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


def conv_creator(netin, config):
    return conv_concat(netin, config['filters'], config['compression'], config['kernel_size'],
                       config['strides'],
                       config['alpha'])


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


def upsample_creator(netin, config):
    return upsample(netin, config['size'])


def upsample(netin, size):
    return UpSampling2D(size)(netin)


def crop_creator(netin, config):
    return crop(netin, config['top'], config['bottom'], config['left'], config['right'])


def crop(netin, top, bottom, left, right):
    return Cropping2D(((top, bottom), (left, right)))(netin)


layers = {'conv_leaky': conv_leaky_creator,
          'dconv': dconv_creator,
          'bottleneck_conv': bottleneck_conv_creator,
          'bottleneck_conv_residual': bottleneck_conv_residual_creator,
          'bottleneck_dconv': bottleneck_dconv_creator,
          'bottleneck_dconv_residual': bottleneck_dconv_residual_creator,
          'max_pool': max_pool_creator,
          'avg_pool': avg_pool_creator,
          'time_dist_conv_leaky': time_dist_conv_leaky_creator,
          'time_dist_max_pool': time_dist_max_pool_creator,
          'wr_basic_conv_leaky': wr_basic_conv_leaky_creator,
          'wr_bottleneck_conv_leaky': wr_bottleneck_conv_leaky_creator,
          'wr_inception_conv_leaky': wr_inception_conv_leaky_creator,
          'conv_concat': conv_creator,
          'upsample': upsample_creator,
          'crop': crop_creator
          }
