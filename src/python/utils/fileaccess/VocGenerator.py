import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import load_file
from utils.imageprocessing.Backend import imread
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel


class VocGenerator(DatasetGenerator):
    @property
    def color_format(self):
        return self._color_format

    @property
    def source_dir(self):
        return self.directory

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def n_samples(self):
        return self.__n_samples

    def __init__(self, dir_voc="resource/ext/backgrounds/VOCdevkit/",
                 batch_size: int = 10, shuffle: bool = True, n_samples=None, start_idx=0,
                 classes=None, frac_emtpy=0.1):
        """
        Generator for Pascal VOC Dataset
        :param dir_voc: directory of VOCdevkit
        :param batch_size: size of one batch
        :param shuffle: shuffle set after each epoch or not
        :param n_samples: number of samples to use defaults to all
        :param start_idx: start index for files to use
        :param classes: which classes to use other images will not be load
        :param frac_emtpy: max amount of images without objects
        """

        self.frac_empty = frac_emtpy
        if classes is None:
            classes = [
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
        self.classes = classes
        self._color_format = 'bgr'
        self.shuffle = shuffle
        dir_voc2012 = dir_voc + "VOC2012/"
        dir_voc2007 = dir_voc + "VOC2007/"

        self.directory = [dir_voc2012, dir_voc2007]
        self.__batch_size = batch_size
        imgs_2007 = dir_voc2007 + "JPEGImages/"
        imgs_2012 = dir_voc2012 + "JPEGImages/"

        labels_2007 = dir_voc2007 + "Annotations/"
        labels_2012 = dir_voc2012 + "Annotations/"

        set_2007_train = load_file(dir_voc2007 + "ImageSets/Main/train.txt").split('\n')
        set_2007_train = [(name, labels_2007, imgs_2007) for name in set_2007_train if len(name) > 2]

        set_2007_valid = load_file(dir_voc2007 + "ImageSets/Main/val.txt").split('\n')
        set_2007_valid = [(name, labels_2007, imgs_2007) for name in set_2007_valid if len(name) > 2]

        set_2012_train = load_file(dir_voc2012 + "ImageSets/Main/train.txt").split('\n')
        set_2012_train = [(name, labels_2012, imgs_2012) for name in set_2012_train if len(name) > 2]

        set_2012_valid = load_file(dir_voc2012 + "ImageSets/Main/val.txt").split('\n')
        set_2012_valid = [(name, labels_2012, imgs_2012) for name in set_2012_valid if len(name) > 2]

        set_2007_test = load_file(dir_voc2007 + "ImageSets/Main/test.txt").split('\n')
        set_2007_test = [(name, labels_2007, imgs_2007) for name in set_2007_test if len(name) > 2]

        self.files = set_2007_train + set_2007_valid + set_2012_train + set_2007_test
        self.test_files = set_2012_valid

        if n_samples is not None:
            self.files = self.files[start_idx:n_samples]
        else:
            self.files = self.files[start_idx:]

        if shuffle:
            random.shuffle(self.files)
            random.shuffle(self.test_files)

        self.files = self._filter_classes(self.files)
        self.test_files = self._filter_classes(self.test_files)

        print('VOCGenerator::{} samples found. Using {} for training and {} for testing'.format(len(self.files),
                                                                                                len(self.files),
                                                                                                len(self.test_files)))

        self.__n_samples = len(self.files)

        ObjectLabel.classes = classes.copy()

    def generate(self):
        return self._generate(self.files)

    def generate_valid(self):
        if not self.test_files:
            print("No valid fraction for test files specified")
        return self._generate(self.test_files)

    def _filter_classes(self, files):
        files_filtered = []
        n_empty = 0
        n_empty_max = int(np.ceil(self.batch_size * self.frac_empty))
        for name, label_path, img_path in files:

            with open(label_path + name + '.xml', 'rb') as f:
                tree = ET.parse(f)
                objects = []
                for element in tree.findall('object'):
                    class_name = element.find('name').text
                    objects.append(class_name)

            objects_filtered = [l for l in objects if l in self.classes]
            if objects_filtered:
                files_filtered.append((name, label_path, img_path))
            elif n_empty < n_empty_max:
                files_filtered.append((name, label_path, img_path))
                n_empty += 1
        return files_filtered

    def _generate(self, files):
        current_batch = []
        files_it = iter(files)

        while True:
            try:
                name, label_path, img_path = next(files_it)
                label_file = label_path + name + '.xml'
                img, label, img_file = self._load_sample(label_file, img_path)

                objects_filtered = [l for l in label.objects if l.class_name in self.classes]

                current_batch.append((img, ImgLabel(objects_filtered), img_file))

                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
            except StopIteration:
                if self.shuffle:
                    random.shuffle(files)

                files_it = iter(files)

    def _load_sample(self, path, img_directory) -> (Image, ImgLabel):
        with open(path, 'rb') as f:
            tree = ET.parse(f)
            objects = []
            img_filename = tree.find('filename').text
            img_file = img_directory + img_filename
            img = imread(img_file, color_format=self.color_format)
            for element in tree.findall('object'):
                name = element.find('name').text
                box = element.find('bndbox')
                xmin = int(np.round(float(box.find('xmin').text)))
                ymin = int(np.round(float(box.find('ymin').text)))
                xmax = int(np.round(float(box.find('xmax').text)))
                ymax = int(np.round(float(box.find('ymax').text)))
                objects.append(ObjectLabel(name, np.array([(xmin, img.shape[0] - ymax), (xmax, img.shape[0] - ymin)])))
            return img, ImgLabel(objects), img_filename
