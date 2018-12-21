import copy

import numpy as np

from utils.imageprocessing.Backend import scale_hsv
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformHSV(ImgTransform):
    def __init__(self, h: float, s: float, v: float):
        self.s = s
        self.h = h
        self.v = v

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug = scale_hsv(img_aug, np.array([self.h, self.s, self.v]))

        return img_aug, label_aug
