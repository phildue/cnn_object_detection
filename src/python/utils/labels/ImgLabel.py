from utils.labels.ObjectLabel import ObjectLabel
import copy


class ImgLabel:
    def __init__(self, objects: [ObjectLabel]):
        self.objects = objects

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        rep = 'Objects:\n'
        for obj in self.objects:
            rep += str(obj) + '\n'
        return rep
