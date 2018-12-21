import numpy as np

from modelzoo.Encoder import Encoder
from utils.imageprocessing.Image import Image
from utils.imageprocessing.Imageprocessing import show
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class Preprocessor:
    def __init__(self, encoder: Encoder, n_classes, img_shape, color_format, augmentation: [ImgTransform]=None,
                 preprocessing: [ImgTransform] = None, show_t=-1):
        self.show_t = show_t
        self.preprocessing = preprocessing
        self.color_format = color_format
        self.img_height, self.img_width = img_shape[:2]
        self.n_classes = n_classes
        self.encoder = encoder
        self.augmenter = augmentation

    def preprocess_train_generator(self, batches: [[(Image, ImgLabel)]]):
        for batch in batches:
            yield self.preprocess_train(batch)

    def preproces_test_generator(self, batches: [[(Image, ImgLabel)]]):
        for batch in batches:
            yield self.preprocess_test(batch)

    def preprocess_test(self, dataset: [(Image, ImgLabel)]) -> (np.array, np.array):
        y_batch = []
        x_batch = []
        for img, label, _ in dataset:

            if self.preprocessing is not None:
                for p in self.preprocessing:
                    img, label = p.transform(img, label)

            if self.color_format is not img.format:
                raise ValueError("Invalid Color Format")

            if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
                raise ValueError('Invalid Input Size')

            if self.show_t > -1:
                show(img,labels=label, t=0)

            img_enc = self.encoder.encode_img(img)
            label_enc = self.encoder.encode_label(label)
            label_enc = np.expand_dims(label_enc, 0)
            x_batch.append(img_enc)
            y_batch.append(label_enc)

        y_batch = np.concatenate(y_batch, 0)
        x_batch = np.concatenate(x_batch, 0)
        return x_batch, y_batch

    def preprocess_train(self, dataset: [(Image, ImgLabel)]) -> (np.array, np.array):
        dataset_augmented = []

        if self.augmenter is not None:
            for img, label, path in dataset:
                for a in self.augmenter:
                    img, label = a.transform(img, label)
                dataset_augmented.append((img, label, path))
        else:
            dataset_augmented = dataset

        return self.preprocess_test(dataset_augmented)

    def preprocess(self, img: Image):

        if self.preprocessing is not None:
            for p in self.preprocessing:
                img, _ = p.transform(img, ImgLabel([]))

        if self.color_format != img.format:
            raise ValueError("Invalid Color Format")

        if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
            raise ValueError('Invalid Input Size')

        return self.encoder.encode_img(img)

    def preprocess_batch(self, batch: [Image]):
        x_batch = np.zeros((len(batch), self.img_height, self.img_width, 3))
        for i, img in enumerate(batch):
            x_batch[i] = self.preprocess(img)
        return x_batch

