import numpy as np

from etc.mulitply_adds import count_operations
from visuals import BaseBarPlot


def plot_ops(architecture, volume):
    lookup = count_operations(architecture, volume)

    operations = [entry['operations'] for entry in lookup]
    print('Total Operations:', sum(operations))
    layers = np.arange(0, len(lookup))

    plot = BaseBarPlot(x_data=[layers],
                       y_data=[operations],
                       y_label='Number of Multiply Adds',
                       x_label='Layer',
                       colors=['blue'],
                       title='',
                       width=0.5)
    return plot


def plot_ops_list(architecture, volume, names=None):
    operationss = []
    layerss = []
    for arch in architecture:
        lookup = count_operations(arch, volume)

        operations = [entry['operations'] for entry in lookup]
        print('Total Operations:', sum(operations))
        layers = np.arange(0, len(lookup))
        operationss.append(operations)
        layerss.append(layers)

    plot = BaseBarPlot(x_data=layerss,
                       y_data=operationss,
                       y_label='Number of Multiply Adds',
                       x_label='Layer',
                       legend=names,
                       colors=['blue', 'green', 'red', 'yellow'],
                       title='',
                       width=1 / (1 + len(architecture)))
    return plot


if __name__ == '__main__':
    networks = [
        [
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'max_pool', 'size': (2, 2)},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 30, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (3, 3), 'filters': 30, 'strides': (1, 1), 'alpha': 0.1},

        ],
        # [
        #     {'name': 'dconv', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
        #     {'name': 'max_pool', 'size': (2, 2)},
        #
        # ],
        # [
        #     {'name': 'bottleneck_dconv', 'kernel_size': (3, 3), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1,
        #      'expand': 3},
        #     {'name': 'max_pool', 'size': (2, 2)},
        #
        # ],
        [
            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 16, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 16, 'strides': (2, 2), 'alpha': 0.1},

            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 32, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 32, 'strides': (2, 2), 'alpha': 0.1},

            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},

            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},

            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1},
            {'name': 'conv_leaky', 'kernel_size': (2, 2), 'filters': 64, 'strides': (2, 2), 'alpha': 0.1},

            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},
            {'name': 'bottleneck_conv', 'kernel_size': (2, 2), 'filters': 64, 'strides': (1, 1), 'alpha': 0.1,
             'compression': 1.0},

        ]
    ]
    img = (416, 416, 3)

    plot = plot_ops_list(networks, img, ['Standard Conv',
                                         # 'Depthwise Conv',
                                         # 'Bottleneck DConv',
                                         'Bottleneck Conv'])
    plot.show()
