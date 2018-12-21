import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon
from utils.labels.utils import resize_label

bb = Polygon(np.array([[0, 0],
                       [0, 416],
                       [416, 416],
                       [416, 0]]))

print(bb)
img = Image(np.zeros((416, 416, 3)), 'bgr')

box_sizes = [0.001, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

aspect_ratios = [4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.25]
for b in box_sizes:
    label = ImgLabel([ObjectLabel('box', 1.0, bb)])
    w = np.sqrt(b *  2*416 ** 2 / 2)

    label = resize_label(label, scale_x=w/416, scale_y=w/416)
    print(w)
    print(b)
    print(label.objects[0].poly.area)
    print(label.objects[0].poly.area / (416 * 416))
    show(img, labels=label)

for a in aspect_ratios:
    label = ImgLabel([ObjectLabel('box', 1.0, bb)])
    label = resize_label(label, scale_x=0.1, scale_y=0.1)
    label = resize_label(label, scale_x=1.0, scale_y=a)
    print(label.objects[0].poly.height / label.objects[0].poly.width)
    show(img, labels=label)
