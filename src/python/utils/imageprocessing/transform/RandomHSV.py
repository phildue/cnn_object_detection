import random

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.imageprocessing.transform.TransformHSV import TransformHSV
from utils.labels.ImgLabel import ImgLabel


class RandomHSV(ImgTransform):
    def __init__(self, h: (float, float), v: (float, float), s: (float, float)):
        self.v = v
        self.h = h
        self.s = s

    def transform(self, img: Image, label: ImgLabel):
        scale_h = random.uniform(self.h[0], self.h[1])
        scale_s = random.uniform(self.s[0], self.s[1])
        scale_v = random.uniform(self.v[0], self.v[1])

        img_aug, label_aug = TransformHSV(scale_h, scale_s, scale_v).transform(img, label)

        return img_aug, label_aug
