import unittest

from tests.optimizer_test import SgdOptimizerTestCase

from tests.loss_test import LossTestCase
from tests.layer_test import LayerTestCase
#

if __name__ == '__main__':
    test = SgdOptimizerTestCase()
    test.test_step()