# https://stackoverflow.com/questions/35582521/how-to-calculate-receptive-field-size?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
# Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)
import math
imsize = 416

architecture = [
     {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 128, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 256, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (2, 2)},
    {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 512, 'strides': (1, 1), 'alpha': 0.1},
    # {'name': 'max_pool', 'size': (2, 2)},
    # {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 1024, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'max_pool', 'size': (4, 4)},
    {'name': 'conv_leaky', 'kernel_size': (1, 1), 'filters': 256, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'conv_leaky', 'kernel_size': (5, 5), 'filters': 512, 'strides': (1, 1), 'alpha': 0.1},
    {'name': 'predict'},
]

def arch2dict(arch, upto=None):
    param_dict = {}
    for j, layer in enumerate(arch):
        if layer['name'] == 'conv_leaky':
            param_dict['{0:02d}'.format(j)] = [layer['kernel_size'][0], layer['strides'][0],
                                               (layer['kernel_size'][0] - 1) / 2]
        elif layer['name'] == 'max_pool' or layer['name'] == 'avg_pool':
            try :
                s = layer['strides'][1]
            except KeyError:
                s = layer['size'][1]
            param_dict['{0:02d}'.format(j)] = [layer['size'][0], s, layer['size'][0] / 2]
        elif layer['name'] == 'upsample' or layer['name'] == 'route':
            raise ValueError('Unknown Layer')
        else:
            print("Not handled:")
            print(layer)
        if upto is not None and j > upto: break
    return param_dict


net = arch2dict(architecture)
# net = OrderedDict(sorted(net.items()))
# layers = list(net.values())
# layer_names = list(net.keys())



def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
        layer[0], layer[1], layer[2], layer[3]))


layerInfos = []
if __name__ == '__main__':
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for k in sorted(net.keys()):
        layer = net[k]
        currentLayer = outFromIn(layer, currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, k)

    print("------------------------")
