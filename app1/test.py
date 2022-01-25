import unittest

from tests.optimizer_test import SgdOptimizerTestCase

from tests.loss_test import LossTestCase
from tests.layer_test import LayerTestCase
#

if __name__ == '__main__':
    test = LossTestCase()
    test.test_cross_entropy_loss()
