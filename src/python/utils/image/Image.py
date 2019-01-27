class Image:
    def __init__(self, array, color_format, path='./'):
        self.path = path
        self.format = color_format
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    def copy(self):
        return Image(self.array.copy(), self.format)
