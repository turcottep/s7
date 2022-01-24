import numpy as np
from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    def calculate(self, x, target):
        new_x = softmax(x)
        new_target = np.zeros_like(new_x)
        new_target[np.arange(len(new_x)), target] = 1
        self.loss = np.sum(-np.sum(new_target * np.log(new_x+1e-8), axis=1))
        input_grad = new_x - new_target
        return self.loss, input_grad


def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator, axis=1, keepdims=True) + 1e-8
    new_denominator = np.tile(denominator, (1, numerator.shape[1]))
    new_x = numerator / new_denominator
    return new_x


class MeanSquaredErrorLoss(Loss):
    def calculate(self, x, target):
        self.loss = np.mean(np.square(x - target))
        input_grad = (x - target) / x.shape[1]
        return self.loss, input_grad
