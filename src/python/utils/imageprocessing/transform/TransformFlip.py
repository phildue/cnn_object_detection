import copy

from utils.imageprocessing.Backend import flip
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformFlip(ImgTransform):
    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)
        img_aug, label_aug = flip(img_aug, label_aug, 1)

        return img_aug, label_aug
