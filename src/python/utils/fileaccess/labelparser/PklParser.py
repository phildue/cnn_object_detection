import glob
import pickle

from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing import Image
from utils.imageprocessing.Backend import imwrite, imread
from utils.labels.ImgLabel import ImgLabel


class PklParser(AbstractDatasetParser):

    def write_label(self, label: [ImgLabel], path: str):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

    def read(self, n=0) -> ([Image], [ImgLabel]):
        files = sorted(glob.glob(self.directory + '/*.pkl'))
        samples = []
        labels = []
        for i, file in enumerate(files):
            if 0 < n < i: break
            label = PklParser.read_label(file)
            img = imread(file.replace('pkl', self.image_format), self.color_format)
            samples.append(img)
            labels.append(label)
        return samples, labels

    @staticmethod
    def read_label(filepath: str) -> [ImgLabel]:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
