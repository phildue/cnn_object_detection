from modelzoo.iou import non_max_suppression
from modelzoo.yolo.YoloV3Decoder import YoloV3Decoder
from modelzoo.yolo.architectures import build_yolov3_tiny
from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.image.imageprocessing import show, COLOR_BLUE, COLOR_GREEN
from utils.workdir import cd_work
import numpy as np

cd_work()
dataDir = 'lib/cocoapi/'
batch_size = 10
generator = CocoGenerator(dataDir)
loader = iter(generator.generate_valid(batch_size=batch_size, n=20))
imgs_raw, labels_raw = next(loader)

n_classes = 80
batch_size = 1
network, anchors = build_yolov3_tiny(n_classes)
print(network.summary())


# network.load_weights('resource/yolov3-tiny-weights.h5',by_name=True)

for i, img in enumerate(imgs_raw):
    output_shapes = network.compute_output_shape((batch_size, img.shape[0], img.shape[1], 3))

    grids = [(shape[1], shape[2]) for shape in output_shapes]

    decoder = YoloV3Decoder(img_size=img.shape[:2], grids=grids, anchor_dims=anchors, n_classes=n_classes)

    y_pred = network.predict(np.expand_dims(img.array, 0), batch_size=1)

    label_predicted = decoder.decode_netout(y_pred)
    #label_predicted.objects = non_max_suppression(label_predicted.objects)
    show(img, labels=[label_predicted, labels_raw[i]], colors=[COLOR_BLUE, COLOR_GREEN])
