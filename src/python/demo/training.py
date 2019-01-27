import numpy as np
from keras.optimizers import Adam
from modelzoo.yolo.YoloV3Encoder import YoloV3Encoder
from modelzoo.yolo.YoloV3Loss import YoloV3Loss
from modelzoo.yolo.architectures import build_yolov3_test

from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.image.imageprocessing import pad_equal_letterbox, resize, show, pad_letterbox
from utils.workdir import cd_work

cd_work()
dataDir = 'lib/cocoapi/'

generator = CocoGenerator(dataDir)

n_samples = 10000

n_classes = 92
batch_size = 8

network, anchors = build_yolov3_test(n_classes)

n_boxes = [len(a) for a in anchors]

loss = YoloV3Loss(n_classes)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

network.compile(optimizer=optimizer, loss=loss.total_loss)


def preprocess(gen):
    for imgs, labels in gen:
        img_size = max(imgs[0].shape[:2])

        # ensure that we don't get odd pooling in the network otherwise we cannot concatenate the tensors
        while True:
            output_shapes = network.compute_output_shape((batch_size, img_size, img_size, 3))

            grids = [(shape[1], shape[2]) for shape in output_shapes]

            if any([any((img_size % g[0] > 0, img_size % g[1] > 0)) for g in grids]):
                img_size += 1
            else:
                break
        # print("Image Size: {}x{} --> {}x{}|{}x{}".format(img_size, img_size, grids[0][0], grids[1][1], grids[1][0],
        #                                                 grids[1][1]))
        encoder = YoloV3Encoder(anchor_dims=anchors, n_classes=n_classes,
                                verbose=0, img_size=(img_size, img_size), grids=grids)

        x = np.zeros((batch_size, img_size, img_size, 3))
        y = [np.zeros((batch_size, grid[0], grid[1], n_boxes[ig], n_classes + 4 + 1)) for ig, grid in
             enumerate(grids)]
        idx = 0
        for img, label in zip(imgs, labels):
            img_, label_ = pad_equal_letterbox(img, label)
            img_, label_ = resize(img_, (img_size, img_size), label=label_)
            # show(img_, labels=label_, t=1)
            x[idx] = img_.array
            y_outs = encoder.encode_label(label_)
            for idx_out, y_out in enumerate(y_outs):
                y[idx_out][idx] = y_out
            idx += 1
        yield x, y


network.fit_generator(generator=preprocess(generator.generate_train(batch_size=batch_size, n=-1)),
                      steps_per_epoch=int(n_samples / batch_size),
                      epochs=10,
                      validation_data=preprocess(generator.generate_valid(batch_size=batch_size, n=-1)),
                      validation_steps=20)
