
import numpy as np
from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, input_count, output_count):
        super().__init__()
        w_xavier = np.array(np.random.normal(0, np.sqrt(
            2/(input_count + output_count)), (output_count, input_count)))
        b_xavier = np.array(np.random.normal(
            0, np.sqrt(2/(output_count)), output_count))
        self.parameters = {"w": w_xavier, "b": b_xavier}

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return {"None": np.array([0])}

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

    def _forward_training(self, x):
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        e = self._buffers["epsilon"]
        x_hat = (x - mean) / np.sqrt(variance + e)
        gamma = self._parameters["gamma"]
        beta = self._parameters["beta"]
        y = gamma * x_hat + beta
        cache = [x, mean, variance, x_hat]

        alpha = self._buffers["alpha"]
        self._buffers["global_mean"] = (
            1-alpha)*self._buffers["global_mean"] + alpha*mean
        self._buffers["global_variance"] = (
            1-alpha)*self._buffers["global_variance"] + alpha*variance
        return y, cache

    def _forward_evaluation(self, x):
        mean = self._buffers["global_mean"]
        variance = self._buffers["global_variance"]
        e = self._buffers["epsilon"]
        x_hat = (x - mean) / np.sqrt(variance + e)
        gamma = self._parameters["gamma"]
        beta = self._parameters["beta"]
        y = gamma * x_hat + beta
        cache = [x, mean, variance, x_hat]
        return y, cache

    def backward(self, output_grad, cache):
        gamma = self._parameters["gamma"]
        e = self._buffers["epsilon"]
        [x, mean, variance, x_hat] = cache
        M = x.shape[0]

        d_x_hat = output_grad * gamma  # 67

        d_variance = np.sum(np.multiply(np.multiply(
            d_x_hat, (x-mean)), (-1/2)*np.power((variance+e), (-3/2))), axis=0)  # 68

        d_mean = (-np.sum(d_x_hat, axis=0) /
                  np.sqrt(variance+e)) + (-2/M)*d_variance*np.sum(x-mean, axis=0)  # 69

        d_x = d_x_hat/np.sqrt(variance+e) + (2/M) * \
            d_variance*(x-mean) + (1/M)*d_mean  # 70

        d_gamma = np.sum(output_grad * x_hat, axis=0)  # 71
        d_beta = np.sum(output_grad, axis=0)  # 71

        return d_x, {"gamma": d_gamma, "beta": d_beta}


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
