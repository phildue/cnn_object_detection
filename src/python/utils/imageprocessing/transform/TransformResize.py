import copy

from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformResize(ImgTransform):
    def __init__(self, shape):
        self.shape = shape

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug, label_aug = resize(img_aug, label=label_aug, shape=self.shape)

        return img_aug, label_aug
