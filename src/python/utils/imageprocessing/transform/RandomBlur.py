from utils.imageprocessing.Backend import blur
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel
import numpy as np


class RandomBlur(ImgTransform):
    def __init__(self, kernel=(5, 5), it_min=0, it_max=10):
        self.iterations = np.random.randint(it_min, it_max)
        self.kernel = kernel

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        img_aug = blur(img_aug, self.kernel, self.iterations)
        return img_aug, label
