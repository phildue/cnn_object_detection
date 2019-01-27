from modelzoo.YoloV3Decoder import YoloV3Decoder
from modelzoo.YoloV3Encoder import YoloV3Encoder
from modelzoo.metrics.YoloV3Loss import YoloV3Loss
from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.image.Backend import resize
from utils.image.Imageprocessing import show
from utils.workdir import cd_work
import numpy as np

cd_work()
dataDir = 'lib/cocoapi/'

generator = CocoGenerator(dataDir)
anchors = np.array([
    [[10, 13],
     [16, 30],
     [33, 23]],

    [[30, 61],
     [62, 45],
     [59, 119]],

    [[116, 90],
     [156, 198],
     [373, 326]]])

img_size = (416, 416)

grids = [
    [26, 26],
    [13, 13],
    [7, 7]
]
n_classes = 92
batch_size = 10
encoder = YoloV3Encoder(anchor_dims=anchors, grids=grids, n_classes=n_classes,
                        verbose=0,img_size=img_size)
decoder = YoloV3Decoder(grids=grids, anchor_dims=anchors, n_classes=n_classes, img_size=img_size)

loader = iter(generator.generate_valid(batch_size=batch_size, n=20))
imgs_raw, labels_raw = next(loader)
imgs = []
labels = []
for i in range(len(imgs_raw)):
    img, label = resize(imgs_raw[i], img_size, label=labels_raw[i])
    imgs.append(img)
    labels.append(label)

y_batch = encoder.encode_label_batch(labels)

loss = YoloV3Loss(n_classes)
import keras.backend as K
import tensorflow as tf


def logit(x):
    eps = K.epsilon()
    _x = np.clip(x, eps, 1 - eps)
    y = np.log(_x / (1 - _x))
    return y


def log(x):
    y = np.log(x)
    return y


graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        for i in range(batch_size - 1):

            class_loss = 0
            loc_loss = 0
            conf_loss = 0

            y_enc = []
            for idx_out in range(3):
                y_0 = y_batch[idx_out][i:i + 1].copy()
                y_1 = y_batch[idx_out][i + 1:i + 2].copy()
                y_1 = y_0.copy()
                y_1[:, :, :, :, 0:1] = logit(y_1[:, :, :, :, 0:1])
                y_1[:, :, :, :, 1:n_classes + 1] = np.clip(y_1[:, :, :, :, 1:n_classes + 1], K.epsilon(),
                                                           1 - K.epsilon())
                y_1[:, :, :, :, 1:n_classes + 1] = np.log(y_1[:, :, :, :, 1:n_classes + 1])

                y_1[:, :, :, :, n_classes + 1: n_classes + 1 + 2] = logit(
                    y_1[:, :, :, :, n_classes + 1: n_classes + 1 + 2])
                y_1[:, :, :, :, n_classes + 1 + 2: n_classes + 1 + 4] = log(
                    y_1[:, :, :, :, n_classes + 1 + 2: n_classes + 1 + 4])
                y_enc.append(y_1)
                y_0_t = K.constant(y_0, shape=y_0.shape)
                y_1_t = K.constant(y_1, shape=y_1.shape)

                class_loss += loss.classification_loss(y_0_t, y_1_t).eval()
                loc_loss += loss.localization_loss(y_0_t, y_1_t).eval()
                conf_loss += loss.confidence_loss(y_0_t, y_1_t).eval()

            print('Class Loss: {} - Loc Loss: {} - Conf Loss: {}'.format(class_loss, loc_loss, conf_loss))
            l = decoder.decode_netout_batch(y_enc)[0]
            l.objects = [o for o in l.objects if o.confidence > 0.01]
            show(imgs[i], labels=l)
