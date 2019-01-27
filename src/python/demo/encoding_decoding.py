from modelzoo.YoloV3Decoder import YoloV3Decoder
from modelzoo.YoloV3Encoder import YoloV3Encoder
from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.imageprocessing.Imageprocessing import show
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
n_classes = 85

encoder = YoloV3Encoder(anchor_dims=anchors, img_size=img_size, grids=grids, n_classes=n_classes,
                        verbose=0)
decoder = YoloV3Decoder(img_size=img_size, grids=grids, anchor_dims=anchors, n_classes=n_classes)

loader = iter(generator.generate_valid(batch_size=10, n=20))
imgs, labels = next(loader)

y = encoder.encode_label_batch(labels)

print(y)
# labels_decoded = decoder.decode_netout_batch(y)
#
# for i, l in enumerate(labels_decoded):
#     l.objects = [o for o in l.objects if o.confidence > 0.0]
#     show(imgs[i], labels=l)
