import numpy as np

from utils.fileaccess.utils import load_file


class ModelSummary:

    @staticmethod
    def from_file(path):
        summary = load_file(path)
        try:
            return ModelSummary(summary['architecture'], summary['weights'], summary['img_res'], summary['anchors'],
                            summary['color_format'])
        except KeyError:
            return ModelSummary(summary['architecture'], summary['weights'], 0, summary['anchors'],
                                summary['color_format'])
    def __init__(self, arch, weights, img_res, anchors, color_format):
        # {'model': self.predictor.net.__class__.__name__,
        #  'resolution': self.predictor.input_shape,
        #  'train_params': self.predictor.net.train_params,
        #  'image_source': self.dataset_gen.source_dir,
        #  'color_format': self.dataset_gen.color_format,
        #  'batch_size': self.dataset_gen.batch_size,
        #  'n_samples': self.dataset_gen.n_samples,
        #  'transform': augmentation,
        #  'initial_epoch': self.initial_epoch,
        #  'epochs': self.epochs,
        #  'architecture': self.predictor.net.backend.get_config(),
        # 'weights': self.predictor.net.backend.count_params()}
        self.color_format = color_format
        self.anchors = anchors
        self.img_res = img_res
        self.weights = weights
        self.architecture = arch

    @property
    def img_size(self):
        return self.img_res[0] * self.img_res[1]

    @property
    def max_depth(self):
        d = 0
        outs = []
        for i in range(len(self.architecture)):
            layer = self.architecture[i]
            if 'conv' in layer['name']:
                d += 1
            elif 'predict' in layer['name']:
                outs.append(d)
            elif 'route' in layer['name']:
                idxs = layer['index']
                idxs_abs = []
                for idx in idxs:
                    if idx > 0:
                        idxs_abs.append(d - idx)
                    else:
                        idxs_abs.append(idx)

                for r in range(np.abs(np.min(idxs_abs))):
                    layer_route = self.architecture[i - r]
                    if 'conv' in layer_route['name']:
                        d -= 1
        return np.max(outs)
