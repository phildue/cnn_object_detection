from utils.imageprocessing.Backend import blur
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformerBlur(ImgTransform):
    def __init__(self, kernel=(5, 5), iterations=1):
        self.iterations = iterations
        self.kernel = kernel

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        img_aug = blur(img_aug, self.kernel, self.iterations)
        return img_aug, label
