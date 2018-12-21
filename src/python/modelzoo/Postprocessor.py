from modelzoo.Decoder import Decoder
from modelzoo.iou import non_max_suppression

from utils.labels.ObjectLabel import ObjectLabel


class Postprocessor:
    def __init__(self,
                 decoder: Decoder,
                 iou_thresh=0.4,
                 ):
        self.decoder = decoder
        self.iou_thresh = iou_thresh

    def postprocess(self, netout):
        labels = []
        for i in range(netout.shape[0]):
            label = self.decoder.decode_netout(netout[i])
            objs = self.non_max_suppression(label.objects, self.iou_thresh)
            label.objects = objs
            labels.append(label)
        if len(labels) > 1:
            return labels
        else:
            return labels[0]

    @staticmethod
    def non_max_suppression(boxes: [ObjectLabel], iou_thresh):
        return non_max_suppression(boxes, iou_thresh, n_max=100)
