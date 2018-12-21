import glob
import os
import random

import numpy as np

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.labelparser.DatasetParser import DatasetParser
from utils.imageprocessing.Backend import imread
from utils.labels.ImgLabel import ImgLabel


class GateGenerator(DatasetGenerator):
    @property
    def color_format(self):
        return self._color_format

    @property
    def source_dir(self):
        return self.directories

    @property
    def batch_size(self):
        return self.__batch_size

    def __init__(self, directories: [str], batch_size: int, shuffle: bool = True, img_format: str = 'jpg',
                 color_format='yuv',
                 label_format: str = 'pkl', n_samples=None, valid_frac=0.0, start_idx=0, org_aspect_ratio=1.05,
                 filter=None, remove_filtered=True, max_empty=1.0, forever=True, subsets: [int] = None):
        self.forever = forever
        self._filter = filter
        self.valid_frac = valid_frac
        self.remove_filtered = remove_filtered
        self.org_aspect_ratio = org_aspect_ratio
        self._color_format = color_format
        self.label_format = label_format
        self.__batch_size = batch_size
        self.shuffle = shuffle
        self.img_format = img_format
        self.directories = directories
        self.max_empty_frac = max_empty
        files_all = []
        for i, d in enumerate(directories):
            files_dir = sorted(glob.glob(d + "/*." + img_format))
            if len(files_dir) == 0:
                raise ValueError("No files found in: ", d)
            files_dir = [os.path.abspath(f) for f in files_dir]

            if subsets is not None:
                n = len(files_dir)
                n_subset = int(subsets[i] * n)
                print("From {} selecting {}/{}".format(d, n_subset, n))
                if 1 % subsets[i]:
                    files_dir = np.random.choice(files_dir, n_subset, False)
                else:
                    files_dir = files_dir[::int(1/subsets[i])]

            files_all.extend(files_dir)

        files = files_all[start_idx:]
        if shuffle:
            random.shuffle(files)
        if n_samples is not None:
            files = files[:n_samples]

        print('Gate Generator::{0:d} samples found. Using {1:d}'.format(len(files_all), len(files)))

        self.train_files = files[:int(np.ceil((1 - valid_frac) * len(files)))]
        self.test_files = files[:int(np.floor(valid_frac * len(files)))]
        self.__n_samples = len(self.train_files)

    @property
    def n_samples(self):
        return self.__n_samples

    def __len__(self):
        return self.n_samples

    def generate(self):
        return self._generate(self.train_files)

    def generate_valid(self):
        return self._generate(self.test_files)

    def _generate(self, files):
        current_batch = []
        files_it = iter(files)
        max_empty = int(self.max_empty_frac * self.n_samples)
        n_empty = 0
        if not files:
            raise ValueError('GateGenerator::Cannot generate from empty list.')
        while True:
            try:
                file = next(files_it)
                img = imread(file, self.color_format)
                if self.label_format is not None:
                    label = DatasetParser.read_label(self.label_format,
                                                     file.replace(self.img_format, self.label_format))

                    if label is None:
                        continue

                    if self._filter is not None:
                        label_filtered = self._filter(label)
                    else:
                        label_filtered = label

                    if len(label_filtered.objects) == 0:
                        if n_empty < max_empty:
                            n_empty += 1
                        else:
                            continue

                    n_filtered = len(label.objects) - len(label_filtered.objects)
                    # print("Filtered labels: {}".format(n_filtered))
                    if n_filtered > 0 and self.remove_filtered:
                        continue


                else:
                    label_filtered = ImgLabel([])
                current_batch.append((img, label_filtered, file))
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    del current_batch
                    current_batch = []
            except StopIteration:
                if self.forever:
                    if self.shuffle: random.shuffle(files)
                    files_it = iter(files)
                else:
                    break
