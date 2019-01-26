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

encoder = YoloV3Encoder(anchor_dims=anchors, img_size=img_size, grids=grids, n_classes=85,
                        verbose=0)

for imgs, labels in generator.generate_valid(batch_size=10, n=20):
    #img_t = encoder.encode_img_batch(imgs)
    label_t = encoder.encode_label_batch(labels)
    print("Assigned: {} Lost: {}. Ignored Anchors: {}".format(encoder.matched, encoder.unmatched, encoder.ignored))


    #print(label_t)
