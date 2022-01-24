import numpy as np
from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        # print("tryin to update out here in theses shtreets:", parameter_name,
        #       self._parameters[parameter_name].shape, parameter.shape, parameter_grad.shape)
        # if parameter_name != "None":
        # print("tryin to update out here in theses shtreets:",
        #       parameter_name, self._parameters[parameter_name])
        return parameter - self.learning_rate * parameter_grad
