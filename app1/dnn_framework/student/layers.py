
from linecache import cache
from locale import normalize
from re import S
from statistics import variance
from tkinter.messagebox import NO
import numpy as np
from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, input_count, output_count):
        super().__init__()
        w_xavier = np.array(np.random.normal(0, np.sqrt(
            2/(input_count + output_count)), (output_count, input_count)))
        b_xavier = np.array(np.random.normal(
            0, np.sqrt(2/(output_count)), output_count))
        # print("w_xavier:", w_xavier, "b_xavier:", b_xavier)
        self.parameters = {"w": w_xavier, "b": b_xavier}
        self.buffers = {"None": np.zeros(output_count)}

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.buffers

    def forward(self, x):
        self.x = x
        y = np.matmul(x, self.parameters["w"].T)
        big_b = np.tile(self.parameters['b'], (x.shape[0], 1))
        y += big_b
        cache = [x, self.parameters["w"], self.parameters["b"]]
        return y, cache

    def backward(self, out_grad, cache):
        [x, w, b] = cache
        dx = np.matmul(out_grad, w)
        dw = np.matmul(out_grad.T, x)
        db = np.sum(out_grad, axis=0)
        return dx, {"w": dw, "b": db}


class BatchNormalization(Layer):

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self._parameters = {"gamma": np.ones(
            input_count), "beta": np.zeros(input_count)}
        self._buffers = {'global_mean': np.zeros(
            input_count), 'global_variance': np.zeros(input_count), 'epsilon': 1e-8, 'alpha': alpha}

    def get_parameters(self):
        return self._parameters

    def get_buffers(self):
        return self._buffers

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    # def _forward_training(self, x):
    #     mean = np.mean(x, axis=0)
    #     sigma = np.std(x, axis=0)
    #     variance = np.var(x, axis=0)
    #     alpha = self._buffers["alpha"]
    #     self._buffers['global_mean'] = alpha * \
    #         self._buffers['global_mean'] + (1-alpha)*mean

    #     self._buffers['global_variance'] = alpha * \
    #         self._buffers['global_variance'] + (1-alpha)*variance

    #     x_hat = (x - mean) / np.sqrt(variance + 1e-8)
    #     y = self._parameters["gamma"] * x_hat + self._parameters["beta"]
    #     cache = [x, mean, sigma, x_hat]
    #     return y, cache

    def _forward_training(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.var(x, axis=0)
        e = self._buffers["epsilon"]
        x_norm = (x - mu) / np.sqrt(sigma + e)
        gamma = self._parameters["gamma"]
        beta = self._parameters["beta"]
        y = gamma * x_norm + beta
        cache = [x, mu, sigma, x_norm]

        alpha = self._buffers["alpha"]
        self._buffers["global_mean"] = (
            1-alpha)*self._buffers["global_mean"] + alpha*mu
        self._buffers["global_variance"] = (
            1-alpha)*self._buffers["global_variance"] + alpha*sigma
        return y, cache

    # def _forward_evaluation(self, x):
    #     mean = self.buffers['global_mean']
    #     variance = self.buffers['global_variance']
    #     x_hat = (x - mean) / np.sqrt(variance + 1e-8)
    #     y = self.parameters["gamma"] * x_hat + self.parameters["beta"]
    #     return y, None

    # def backward(self, output_grad, cache):
    #     print("cache:", cache)
    #     gamma = self.parameters["gamma"]
    #     x = self.x
    #     x_hat = self.x_hat
    #     mean = np.mean(x, axis=0)
    #     var = np.var(x, axis=0)

    #     dx_hat = output_grad * gamma  # (67)

    #     d_var = np.sum(dx_hat * (x - mean) * (-0.5)
    #                    * (var + 1e-8)**(-3/2), axis=0)  # (68)

    #     d_mean = -np.sum(dx_hat * (-1/np.sqrt(var + 1e-8)), axis=0) + \
    #         -2/x.shape[0] * d_var * np.sum(x - mean, axis=0)  # (69)

    #     d_x = dx_hat / np.sqrt(var + 1e-8) + d_var * 2/x.shape[0] * \
    #         (x - mean) + d_mean / x.shape[0]  # (70)

    #     d_gamma = np.sum(x_hat * output_grad, axis=0)  # (71)

    #     d_beta = np.sum(output_grad, axis=0)  # (71)

    #     return d_x, {"gamma": d_gamma, "beta": d_beta}

    def _forward_evaluation(self, x):
        mu = self._buffers["global_mean"]
        sigma = self._buffers["global_variance"]
        e = self._buffers["epsilon"]
        x_norm = (x - mu) / np.sqrt(sigma + e)
        gamma = self._parameters["gamma"]
        beta = self._parameters["beta"]
        y = gamma * x_norm + beta
        cache = [x, mu, sigma, x_norm]
        return y, cache

    def backward(self, output_grad, cache):
        gamma = self._parameters["gamma"]
        e = self._buffers["epsilon"]
        M = cache[0].shape[0]

        dLdx_norm = output_grad * gamma
        dLdsigma = np.sum(np.multiply(np.multiply(
            dLdx_norm, (cache[0]-cache[1])), (-1/2)*np.power((cache[2]+e), (-3/2))), axis=0)
        dLdmu = (-np.sum(dLdx_norm, axis=0) /
                 np.sqrt(cache[2]+e)) + (-2/M)*dLdsigma*np.sum(cache[0]-cache[1], axis=0)
        dLdx = dLdx_norm/np.sqrt(cache[2]+e) + (2/M) * \
            dLdsigma*(cache[0]-cache[1]) + (1/M)*dLdmu
        dLdgamma = np.sum(output_grad * cache[3], axis=0)
        dLdbeta = np.sum(output_grad, axis=0)

        dLdparam = {"gamma": dLdgamma,
                    "beta": dLdbeta}

        return dLdx, dLdparam


class Sigmoid(Layer):

    def get_parameters(self):
        return {"None": np.array([0])}

    def get_buffers(self):
        return {"None": np.array([0])}

    def forward(self, x):
        y = 1 / (1 + np.exp(-x) + 1e-8)
        cache = [x, y]
        return y, cache

    def backward(self, output_gradient, cache):
        [x, y] = cache
        parameter_gradient = {"None": np.array([0])}
        return output_gradient * y * (1 - y), parameter_gradient


class ReLU(Layer):

    def get_parameters(self):
        return {"None": np.array([0])}

    def get_buffers(self):
        return {"None": np.array([0])}

    def forward(self, x):
        cache = [x]
        return np.maximum(x, 0), cache

    def backward(self, output_gradient, cache):
        parameter_gradient = {"None": np.array([0])}
        x = cache[0]
        return np.where(x > 0, output_gradient, 0), parameter_gradient
