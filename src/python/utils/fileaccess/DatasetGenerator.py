from abc import ABC, abstractmethod


class DatasetGenerator(ABC):
    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def generate_valid(self):
        pass

    @property
    @abstractmethod
    def n_samples(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def source_dir(self):
        pass

    @property
    @abstractmethod
    def color_format(self):
        pass
