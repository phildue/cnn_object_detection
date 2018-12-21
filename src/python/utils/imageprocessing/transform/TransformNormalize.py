from utils.imageprocessing.Backend import normalize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformNormalize(ImgTransform):
    def transform(self, img: Image, label: ImgLabel):
        return normalize(img), label.copy()
