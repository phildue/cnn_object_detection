from abc import abstractmethod

from utils.imageprocessing import Image
from utils.imageprocessing.Backend import imwrite
from utils.labels.ImgLabel import ImgLabel


class AbstractDatasetParser:

    def __init__(self, directory: str, color_format, start_idx=0, image_format='jpg'):
        self.color_format = color_format
        self.image_format = image_format
        self.directory = directory
        self.idx = start_idx

    def write(self, images: [Image], labels: [ImgLabel]):
        for i, l in enumerate(labels):
            filename = '{}/{:05d}'.format(self.directory, self.idx)
            self.write_img(images[i], filename + '.' + self.image_format)
            self.write_label(l, filename)
            self.idx += 1

    def write_img(self, image: Image, path: str):
        imwrite(image, path)

    def write_label(self, label: ImgLabel, path: str):
        pass

    @abstractmethod
    def read(self, n=0) -> ([Image], [ImgLabel]):
        pass

    @staticmethod
    @abstractmethod
    def read_label(filepath: str) -> [ImgLabel]:
        pass
