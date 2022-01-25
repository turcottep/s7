import os

from tqdm import tqdm

from dnn_framework import Trainer, CrossEntropyLoss, SgdOptimizer, \
    LossMetric, ClassificationAccuracyMetric, LossAccuracyLearningCurves
from mnist.dataset import MnistDataset
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class MnistTrainer(Trainer):
    def __init__(self, network, learning_rate, epoch_count, batch_size, output_path):
        loss = CrossEntropyLoss()
        optimizer = SgdOptimizer(
            network.get_parameters(), learning_rate=learning_rate)

        training_dataset = MnistDataset('training')
        validation_dataset = MnistDataset('validation')
        test_dataset = MnistDataset('testing')

        super().__init__(network, training_dataset, validation_dataset, test_dataset,
                         loss, optimizer,
                         epoch_count, batch_size, output_path)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_metric = ClassificationAccuracyMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()
        self._learning_curves = LossAccuracyLearningCurves()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()

    def _measure_training_metrics(self, loss, network_output, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(network_output, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, network_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(network_output, target)

    def _save_figures(self, output_path):
        self._learning_curves.add_training_loss_value(
            self._training_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(
            self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_loss_value(
            self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(
            self._validation_accuracy_metric.get_accuracy())
        self._learning_curves.save_figure(
            os.path.join(output_path, 'learning_curves.png'))

    def _print_metrics(self):
        print('\nTraining : Loss={}, Accuracy={}'.format(self._training_loss_metric.get_loss(),
                                                         self._training_accuracy_metric.get_accuracy()))
        print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_accuracy_metric.get_accuracy()))
        return [self._training_loss_metric.get_loss(), self._training_accuracy_metric.get_accuracy(), self._validation_loss_metric.get_loss(), self._validation_accuracy_metric.get_accuracy()]

    def _test(self, network, test_dataset_loader):
        test_accuracy_metric = ClassificationAccuracyMetric()

        for x, target in tqdm(test_dataset_loader):
            y = network.forward(x)
            plt.imshow(x[0].reshape((28, 28)), cmap=cm.Greys_r)
            # draw in red the predicted and in blue the true digit
            plt.text(20, 20, str(y[0].argmax()), color='red')
            # print(target)
            plt.text(20, 25, str(target[0]), color='green')
            plt.show()
            test_accuracy_metric.add(y, target)

        print('Accuracy={}'.format(test_accuracy_metric.get_accuracy()))
