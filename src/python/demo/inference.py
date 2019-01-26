from modelzoo.YoloV3Decoder import YoloV3Decoder
from modelzoo.YoloV3Encoder import YoloV3Encoder
from modelzoo.architectures import build_yolov3_tiny
from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.workdir import cd_work

cd_work()
dataDir = 'lib/cocoapi/'

generator = CocoGenerator(dataDir)

n_classes = 92
batch_size = 1
model, grids, anchors, img_size = build_yolov3_tiny(n_classes)

model.load_weights('resource/yolov3-tiny-weights.h5')
print(model.summary())

decoder = YoloV3Decoder(img_size=img_size, grids=grids, anchor_dims=anchors, n_classes=n_classes)

