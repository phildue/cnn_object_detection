from utils.imageprocessing.transform.RandomCrop import RandomCrop
from utils.imageprocessing.transform.RandomEnsemble import RandomEnsemble
from utils.imageprocessing.transform.TransformFlip import TransformFlip


class SSDAugmenter(RandomEnsemble):
    def __init__(self, p_equalize=0.5, p_flip=0.5,
                 p_translate=0.5, p_scale=0.5, p_gray=0.5, p_brightness=0.5, p_crop=0.5):
        augmenters = [(p_flip, TransformFlip()),
                      (p_crop, RandomCrop(c_max=0.9, c_min=0.7))]

        super().__init__(augmenters)
