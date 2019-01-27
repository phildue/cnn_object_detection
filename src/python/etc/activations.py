import numpy as np
from keras import Input, Model

from modelzoo.build_model import build_detector
from modelzoo.layers import create_layer
from utils.image.Backend import imread, imshow, resize
from utils.image.Image import Image
from utils.workdir import cd_work

cd_work()
network = [
    # First layer it does not see complex shapes so we apply a few large filters for efficiency
    {'name': 'conv_leaky', 'kernel_size': (5, 5), 'filters': 16, 'strides': (2, 2),
     'alpha': 0.1},
    # Second layers we use more but smaller filters to combine shapes in non-linear fashion
    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 24, 'strides': (2, 2),
     'alpha': 0.1,
     'compression': 0.5},
    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 30, 'strides': (2, 2),
     'alpha': 0.1,
     'compression': 0.5},
    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 40, 'strides': (2, 2),
     'alpha': 0.1,
     'compression': 0.5},

    {'name': 'bottleneck_conv', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1),
     'alpha': 0.1,
     'compression': 0.5},
    # Final layers, the shapes should be "exhausted" now its about combining spatial information
    # That is why we increase kernel size to collect it
    {'name': 'bottleneck_conv', 'kernel_size': (7, 7), 'filters': 64, 'strides': (1, 1),
     'alpha': 0.1,
     'compression': 0.5},
    {'name': 'predict'},

    {'name': 'bottleneck_conv', 'kernel_size': (9, 9), 'filters': 32, 'strides': (1, 1),
     'alpha': 0.1,
     'compression': 0.5},

    {'name': 'bottleneck_conv', 'kernel_size': (9, 9), 'filters': 24, 'strides': (1, 1),
     'alpha': 0.1,
     'compression': 0.5},
    {'name': 'predict'},

    # receptive field is problematic as the final layer does not see the object
    # if this network works well we should use dilated convolutions in the later layers
]
detector = build_detector(img_shape=(416, 416, 3),
                          anchors=[
                              [[10, 14],
                               [23, 27],
                               [37, 58]],
                              [[81, 82],
                               [135, 169],
                               [344, 319]],
                          ],
                          architecture=network,
                          )
detector.load_weights('out/last_hope/mavnet208x208-jevois/model.h5', )

netin = Input((416, 416, 3))
netout = netin
for i in range(1):
    netout = create_layer(netout, network[i])

model = Model(netin, netout)
model.set_weights(detector.get_weights())
# for i in range(1, 2):
#     model.layers[i].set_weights(weights[i])

img = imread('resource/ext/samples/daylight_course1/00055.jpg', 'bgr')
mat = np.expand_dims(img.array, 0)
netout = model.predict(mat)[0]
netout_scaled = (netout - np.min(netout)) / np.ptp(netout) * 255
imshow(img, 'org')
for i in range(netout_scaled.shape[2]):
    activations = Image(netout_scaled[:, :, i].astype(np.uint8), 'bgr')
    activations = resize(activations, (416, 416))
    imshow(activations, str(i))
