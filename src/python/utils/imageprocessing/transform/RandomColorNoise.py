from random import randint

from utils.imageprocessing.Backend import noisy, noisy_color
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomColorNoise(ImgTransform):
    def __init__(self, variance=10, it_min=0, it_max=10):
        self.iterations = randint(it_min, it_max)
        self.variance = variance

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        img_aug = noisy_color(img_aug, self.variance, self.iterations)
        return img_aug, label
