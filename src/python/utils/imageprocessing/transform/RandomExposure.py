import random

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.imageprocessing.transform.TransformExposure import TransformExposure
from utils.labels.ImgLabel import ImgLabel


class RandomExposure(ImgTransform):
    def __init__(self, e: (float, float)):
        self.e = e

    def transform(self, img: Image, label: ImgLabel):
        e = random.uniform(self.e[0], self.e[1])

        img_aug, label_aug = TransformExposure(1.0, e).transform(img, label)

        return img_aug, label_aug
