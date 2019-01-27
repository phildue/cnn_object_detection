from modelzoo.YoloV3Decoder import YoloV3Decoder
from modelzoo.YoloV3Encoder import YoloV3Encoder
from modelzoo.architectures import build_yolov3_tiny
from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.workdir import cd_work
import numpy as np
cd_work()
dataDir = 'lib/cocoapi/'
batch_size = 1
generator = CocoGenerator(dataDir)
loader = iter(generator.generate_valid(batch_size=batch_size, n=20))
imgs_raw, labels_raw = next(loader)

n_classes = 92
batch_size = 1
model, grids, anchors, img_size = build_yolov3_tiny(n_classes)
print(model.summary())

y_pred = model.predict(np.concatenate([np.expand_dims(i.array,0) for i in imgs_raw],0), batch_size=batch_size)
# model.load_weights('resource/yolov3-tiny-weights.h5')
print(y_pred)
decoder = YoloV3Decoder(img_size=img_size, grids=grids, anchor_dims=anchors, n_classes=n_classes)
