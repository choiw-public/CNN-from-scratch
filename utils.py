import numpy as np


def one_hot_encode(label, num_classes):
    onehot = np.zeros([len(label), num_classes]).astype(np.float32)
    for batch_index, v in enumerate(label):
        onehot[batch_index, v] = 1
    return onehot


def glorot_uniform_init(weight_shape):
    if len(weight_shape) == 4:
        fan_in = np.prod(weight_shape[:-1])
        fan_out = np.prod(weight_shape[:2]) * weight_shape[-1]
    elif len(weight_shape) == 2:
        fan_in = weight_shape[0]
        fan_out = weight_shape[1]
    else:
        raise ValueError('unexpected weight shape')
    expected_min_max = (6.0 / (fan_in + fan_out)) ** 0.5
    return np.random.uniform(-expected_min_max, expected_min_max, weight_shape).astype(np.float32)


def get_out_length(length_in, size, stride):
    return int(np.floor((length_in - size) / stride) + 1)


def get_l2_loss(weight_decay, parameter):
    return 0.5 * weight_decay * np.sum(parameter ** 2)


def get_window(tensor, h_range, w_range, size, stride):
    for i in range(h_range):
        for j in range(w_range):
            h1 = i * stride
            h2 = i * stride + size
            w1 = j * stride
            w2 = j * stride + size
            window = tensor[:, h1:h2, w1:w2, :]
            yield window, i, j, h1, h2, w1, w2


def apply_gradient(weight, accumulation, gradient, args):
    if args['optimizer'] == 'gradient_descent':
        return weight - args['lr'] * (gradient + args['weight_decay'] * weight), None
    elif args['optimizer'] == 'momentum':
        accumulation = 0.9 * accumulation - (gradient + args['weight_decay'] * weight)
        return weight + args['lr'] * accumulation, accumulation
    else:
        raise ValueError('unexpected optimizer')
