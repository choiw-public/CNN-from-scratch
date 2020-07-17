import numpy as np
from private import utils


class Cost:
    def forward(self, x, l2_loss):
        self.shape = x.shape
        self.value = x.sum() / np.array(self.shape[0]).astype(np.float32) + l2_loss

    def backward(self, *dummy):
        self.grads = [np.ones(self.shape).astype(np.float32) * 1.0 / np.array(self.shape[0]).astype(np.float32)]


class CrossEntropy:
    def forward(self, x, arg):
        if x.shape != arg['y'].shape:
            raise ValueError('shape of tensor_in and tensor_gt should be same')
        self.tensor_in = x
        self.tensor_gt = arg['y']
        self.value = np.sum(-arg['y'] * np.log(x), 1)

    def backward(self, gradient, *dummy):
        self.grads = [np.expand_dims(gradient[0].copy(), 1) * (-self.tensor_gt / self.tensor_in)]


class SquaredError:
    def forward(self, x, arg):
        if x.shape != arg['y'].shape:
            raise ValueError('shape of tensor_in and tensor_gt should be same')
        self.tensor_in = x
        self.tensor_gt = arg['y']
        self.value = np.sum((x - arg['y']) ** 2, (1, 2, 3))

    def backward(self, gradient, *dummy):
        dsquarderror = 2 * (self.tensor_in - self.tensor_gt)
        dJ_dsquarderror = np.expand_dims(gradient[0], [1, 2, 3]) * dsquarderror
        self.grads = [dJ_dsquarderror]


class Softmax:
    def forward(self, x, *dummy):
        x = x.reshape([x.shape[0], -1])
        self.tensor_in = x
        exp = np.exp(x)
        self.value = exp / np.expand_dims(np.sum(exp, 1), 1)

    def backward(self, gradient, *dummy):
        batch_num, feature_num = self.tensor_in.shape
        softmax = self.value        
        a = np.zeros([batch_num, feature_num ** 2]).astype(np.float32)
        a[:, ::feature_num + 1] = softmax
        a = a.reshape([batch_num, feature_num, feature_num])

        softmax = np.expand_dims(softmax, 2)
        b = np.tile(softmax, [1, 1, feature_num])
        c = np.transpose(b, [0, 2, 1])
        ds_dx = a - b * c
        dJ_dsoftmax = np.expand_dims(gradient[0].copy(), 1) @ ds_dx
        self.grads = [dJ_dsoftmax.reshape([batch_num, feature_num])]


class ReLU:
    def forward(self, x, *dummy):
        self.tensor_in = x
        self.value = np.maximum(0, x)

    def backward(self, gradient, *dummy):
        self.grads = [gradient[0].copy()]
        self.grads[0][self.tensor_in <= 0] = 0


class FC:
    def __init__(self, in_length, out_length, bias=True):
        self.weight = utils.glorot_uniform_init([in_length, out_length])
        self._dJ_dw_accu = np.zeros_like(self.weight).astype(np.float32)
        if bias:
            self.bias = np.zeros(out_length).astype(np.float32)
            self._dJ_db_accu = np.zeros_like(self.bias).astype(np.float32)
        else:
            self.bias = None

    def forward(self, x, args):
        if len(x.shape) not in [2, 4]:
            raise ValueError('input shape of FC layer should be 2D or 4D')
        if len(x.shape) == 4:
            self.inbound_shape = x.shape
            x = x.reshape(x.shape[0], -1)
        else:
            self.inbound_shape = x.shape
        self.tensor_in = x
        tensor_out = self.tensor_in @ self.weight

        self.l2_loss = utils.get_l2_loss(args['weight_decay'], self.weight)
        if self.bias is not None:
            self.value = tensor_out + self.bias
            self.l2_loss += utils.get_l2_loss(args['weight_decay'], self.bias)
        self.value = tensor_out

    def backward(self, gradient, args):
        dJ_dlayer = gradient[0] @ self.weight.T
        dJ_dlayer = dJ_dlayer.reshape(self.inbound_shape)
        if len(self.tensor_in) == 4:
            dlayer_dw = np.expand_dims(self.tensor_in.reshape(self.tensor_in.shape[0], -1), 2)
        else:
            dlayer_dw = np.expand_dims(self.tensor_in, 2)

        dJ_dw = np.sum(dlayer_dw @ np.expand_dims(gradient[0], 1), 0)
        self.weight, self._dJ_dw_accu = utils.apply_gradient(self.weight, self._dJ_dw_accu, dJ_dw, args)
        self.weight -= args['weight_decay'] * self.weight
        if self.bias is not None:
            dJ_db = gradient[0].sum(0)
            self.bias, self._dJ_db_accu = utils.apply_gradient(self.bias, self._dJ_db_accu, dJ_db, args)
            self.bias -= args['weight_decay'] * self.bias
            self.grads = [dJ_dlayer, dJ_dw, dJ_db]
        else:
            self.grads = [dJ_dlayer, dJ_dw]


class Convolution:
    def __init__(self, filter_size, in_depth, out_depth, stride, bias=True):
        self.out_depth = out_depth
        if filter_size % 2 == 0:
            raise ValueError("this implementation does not consider an even number of 'weight_size'")
        self.stride = stride
        self.weight_size = filter_size
        self.weight = utils.glorot_uniform_init([filter_size, filter_size, in_depth, out_depth])
        self._dJ_dw_accu = np.zeros_like(self.weight).astype(np.float32)
        if bias:
            self.bias = (np.zeros(out_depth)).astype(np.float32)
            self._dJ_db_accu = np.zeros_like(self.bias).astype(np.float32)
        else:
            self.bias = None

    def forward(self, x, args):
        batch, h, w, c = x.shape
        self.h_range = utils.get_out_length(h, self.weight_size, self.stride)
        self.w_range = utils.get_out_length(w, self.weight_size, self.stride)
        self.tensor_in = x
        tensor_out = np.zeros((batch, self.h_range, self.w_range, self.out_depth)).astype(np.float32)
        for window, i, j, _, _, _, _ in utils.get_window(x, self.h_range, self.w_range, self.weight_size, self.stride):
            tensor_out[:, i, j, :] = np.tensordot(window, self.weight, axes=([1, 2, 3], [0, 1, 2]))
        self.l2_loss = utils.get_l2_loss(args['weight_decay'], self.weight)
        if self.bias is not None:
            self.value = tensor_out + self.bias
            self.l2_loss += utils.get_l2_loss(args['weight_decay'], self.bias)
        self.value = tensor_out

    def backward(self, gradient, args):
        f_size = self.weight_size
        dJ_dw = np.zeros(self.weight.shape).astype(np.float32)
        dJ_dlayer = np.zeros(self.tensor_in.shape).astype(np.float32)
        for window, i, j, h1, h2, w1, w2 in utils.get_window(self.tensor_in, self.h_range, self.w_range, f_size, self.stride):
            gradient_window = np.expand_dims(gradient[0][:, i:i + 1, j:j + 1, :], 3)
            dJ_dw += np.sum(np.expand_dims(window, 4) @ gradient_window, 0)
            dJ_dlayer[:, h1:h2, w1:w2, :] += np.sum(gradient_window * np.expand_dims(self.weight, 0), -1)

        # utils.apply_gradient
        self.weight, self._dJ_dw_accu = utils.apply_gradient(self.weight, self._dJ_dw_accu, dJ_dw, args)
        self.weight -= args['weight_decay'] * self.weight

        if self.bias is not None:
            dJ_db = np.sum(gradient[0], (0, 1, 2))
            self.bias, self._dJ_db_accu = utils.apply_gradient(self.bias, self._dJ_db_accu, dJ_db, args)
            self.bias -= args['weight_decay'] * self.bias
            self.grads = [dJ_dlayer, dJ_dw, dJ_db]
        else:
            self.grads = [dJ_dlayer, dJ_dw]


class Pool:
    def __init__(self, pool_size, stride, pooling_type):
        self.pool_size = pool_size
        self.stride = stride
        self.type = pooling_type

    def forward(self, x, *dummy):
        batch, h, w, c = x.shape
        self.tensor_in = x
        self.h_range = utils.get_out_length(h, self.pool_size, self.stride)
        self.w_range = utils.get_out_length(w, self.pool_size, self.stride)
        tensor_out = np.zeros((batch, self.h_range, self.w_range, c)).astype(np.float32)
        for window, i, j, _, _, _, _ in utils.get_window(x, self.h_range, self.w_range, self.pool_size, self.stride):
            if self.type == 'max':
                tensor_out[:, i, j, :] = np.amax(window, axis=(1, 2))
            elif self.type == 'average':
                tensor_out[:, i, j, :] = np.mean(window, axis=(1, 2))
            else:
                raise ValueError('unexpected pooling type')
        self.value = tensor_out

    def backward(self, gradient, *dummy):
        batch, _, _, c = self.tensor_in.shape
        dJ_dlayer = np.zeros_like(self.tensor_in).astype(np.float32)

        if self.type == 'average':
            pooling_kernel = np.ones([batch, self.pool_size, self.pool_size, c]) / self.pool_size ** 2
        for window, i, j, h1, h2, w1, w2 in utils.get_window(self.tensor_in, self.h_range, self.w_range, self.pool_size, self.stride):
            gradient_window = gradient[0][:, i:i + 1, j:j + 1, :]
            if self.type == 'max':
                pooling_kernel = window.reshape([batch, -1, c])
                pooling_kernel = np.array(pooling_kernel.max(axis=1, keepdims=True) == pooling_kernel).astype(np.float32).reshape(window.shape)
            dJ_dlayer[:, h1:h2, w1:w2, :] += gradient_window * pooling_kernel
        self.grads = [dJ_dlayer]


class BatchNormalization:
    def __init__(self):
        self.decay = 0.99
        self.moving_mean = None
        self.moving_var = None
        self.gamma = None
        self.beta = None

    def forward(self, x, args):
        self.tensor_in = x
        if args['is_train']:
            if self.gamma is None:
                self.gamma = np.ones(x.shape[-1]).astype(np.float32)
                self.beta = np.zeros(x.shape[-1]).astype(np.float32)
                self._dJ_dgamma_accu = np.ones(x.shape[-1]).astype(np.float32)
                self._dJ_dbeta_accu = np.ones(x.shape[-1]).astype(np.float32)

            if len(x.shape) == 2:  # in case fullly connected
                self.collapse_axis = 0
            elif len(x.shape) == 4:  # in case convolutoin
                self.collapse_axis = (0, 1, 2)
            else:
                raise ValueError('unexpected x shape')
            # check mean over all collapse_axis or only bath
            self.mean, self.var = np.mean(x, self.collapse_axis), np.var(x, self.collapse_axis)

            if self.moving_mean is None:
                self.moving_mean = np.zeros_like(self.mean)
                self.moving_var = np.ones_like(self.var)

            self.moving_mean = self.decay * self.moving_mean + (1 - self.decay) * self.mean
            self.moving_var = self.decay * self.moving_var + (1 - self.decay) * self.var
            # by following the formula in paper
            self.x_hat = (x - self.mean) / np.sqrt(self.var + 1e-3)
            self.value = self.gamma * self.x_hat + self.beta
        else:
            self.x_hat = (x - self.moving_mean) / np.sqrt(self.moving_var + 1e-3)
            self.value = self.gamma * self.x_hat + self.beta

    def backward(self, gradient, args):
        m = np.prod(self.tensor_in.shape[:-1]).astype(np.float32)
        dJ_dx_hat = gradient[0] * self.gamma
        x_minus_mean = self.tensor_in - self.mean
        std_inv = 1. / np.sqrt(self.var + 1e-3)
        dJ_dvar = np.sum(dJ_dx_hat * x_minus_mean, self.collapse_axis) * -0.5 * std_inv ** 3
        dJ_dmean = np.sum(dJ_dx_hat * -std_inv, self.collapse_axis) + dJ_dvar * np.mean(-2.0 * x_minus_mean,
                                                                                        self.collapse_axis)
        dJ_dx = (dJ_dx_hat * std_inv) + (dJ_dvar * 2 * x_minus_mean / m) + (dJ_dmean / m)
        dJ_dgamma = np.sum(gradient[0] * self.x_hat, self.collapse_axis)
        dJ_dbeta = np.sum(gradient[0], self.collapse_axis)

        self.beta, self._dJ_dbeta_accu = utils.apply_gradient(self.beta, self._dJ_dbeta_accu, dJ_dbeta, args)
        self.gamma, self._dJ_dgamma_accu = utils.apply_gradient(self.gamma, self._dJ_dgamma_accu, dJ_dgamma, args)
        self.grads = [dJ_dx, dJ_dgamma, dJ_dbeta]
