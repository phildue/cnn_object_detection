from keras import Model, Input
from keras.layers import Conv2D, LeakyReLU, Reshape
from keras.optimizers import Adam
from modelzoo.YoloV3Decoder import YoloV3Decoder
from modelzoo.YoloV3Encoder import YoloV3Encoder
from modelzoo.layers.ConcatMeta import ConcatMeta
from modelzoo.metrics.YoloV3Loss import YoloV3Loss
from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Imageprocessing import show
from utils.workdir import cd_work
import numpy as np
import keras.backend as K
import tensorflow as tf

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
    #   [116, 90],
    #   [156, 198],
    #   [373, 326]
])

n_samples = 100

img_size = (52, 52)

grids = [
    #   [26, 26],
    [13, 13],
    [7, 7]
]
n_classes = 92
batch_size = 1
n_boxes = [3,3]
encoder = YoloV3Encoder(anchor_dims=anchors, img_size=img_size, grids=grids, n_classes=n_classes,
                        verbose=0)
decoder = YoloV3Decoder(img_size=img_size, grids=grids, anchor_dims=anchors, n_classes=n_classes)

loss = YoloV3Loss(n_classes)

netin = Input(shape=(img_size[0], img_size[1], 3))
conv1 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='same')(netin)  # 26
act1 = LeakyReLU(0.4)(conv1)
conv2 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv1)  # 13
act2 = LeakyReLU(0.4)(conv2)
out1 = Conv2D(n_boxes[0] * (n_classes + + 1 + 4), kernel_size=(1, 1))(act2)
conv3 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2)  # 13
act3 = LeakyReLU(0.4)(conv3)
out2 = Conv2D(n_boxes[1] * (n_classes + + 1 + 4), kernel_size=(1, 1))(act3)

network = Model(netin, [out1, out2])
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

network.compile(optimizer=optimizer, loss=loss.total_loss)


def preprocess(gen):
    for imgs, labels in gen:
        x = np.zeros((batch_size, img_size[0], img_size[1], 3))
        y = [np.zeros((batch_size, grid[0], grid[1], n_boxes[ig], n_classes + 4 + 1 + 6)) for ig, grid in
             enumerate(grids)]
        idx = 0
        for img, label in zip(imgs, labels):
            img_, label_ = resize(img, img_size, label=label)
            x[idx] = img_.array
            y_outs = encoder.encode_label(label_)
            for idx_out, y_out in enumerate(y_outs):
                y[idx_out][idx] = y_out
            idx += 1
        yield x, y


network.fit_generator(generator=preprocess(generator.generate_train(batch_size=batch_size, n=10)),
                      steps_per_epoch=int(n_samples / batch_size),
                      epochs=10,
                      validation_data=preprocess(generator.generate_valid(batch_size=batch_size, n=10)),
                      validation_steps=20)
