from abc import ABC, abstractmethod

from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class TransformerColorSpace(ABC):
    def __init__(self, out_format: str):
        self.out_format = out_format

    @abstractmethod
    def augment(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        if self.out_format is 'yuv':
            img_aug = img_aug.yuv
        elif self.out_format is 'bgr':
            img_aug = img_aug.bgr
        return img_aug, label.copy()
