import random

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.imageprocessing.transform.TransformChromAbberr import TransformChromAbberr
from utils.labels.ImgLabel import ImgLabel


class RandomChromatic(ImgTransform):
    def __init__(self, r: (float, float), g: (float, float), b: (float, float)):
        self.r = r
        self.g = g
        self.b = b

    def transform(self, img: Image, label: ImgLabel):
        g = random.uniform(self.g[0], self.g[1])
        b_x = random.uniform(self.b[0], self.b[1])
        r_x = random.uniform(self.r[0], self.r[1])
        b_y = random.uniform(self.b[0], self.b[1])
        r_y = random.uniform(self.r[0], self.r[1])

        img_aug, label_aug = TransformChromAbberr(scale=(1.0, g, 1.0), t_x=(b_x, 0, r_x), t_y=(b_y, 0, r_y)).transform(
            img, label)

        return img_aug, label_aug
