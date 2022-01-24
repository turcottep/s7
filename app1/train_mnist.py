import argparse

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist import MnistTrainer


def main():
    # parser = argparse.ArgumentParser(description='Train Backbone')
    # parser.add_argument('--learning_rate', type=float,
    #                     help='Choose the learning rate', required=True)
    # parser.add_argument('--batch_size', type=int,
    #                     help='Set the batch size for the training', required=True)
    # parser.add_argument('--epoch_count', type=int,
    #                     help='Choose the epoch count', required=True)
    # parser.add_argument('--output_path', type=str,
    #                     help='Choose the output path', required=True)

    # parser.add_argument('--checkpoint_path', type=str,
    #                     help='Choose the output path', default=None)

    # args = parser.parse_args()

    checkpoint_path = None
    learning_rate = 0.001
    epoch_count = 12
    batch_size = 1024
    output_path = './output'

    network = create_network(checkpoint_path)
    trainer = MnistTrainer(network, learning_rate,
                           epoch_count, batch_size, output_path)
    trainer.train()


def create_network(checkpoint_path):
    layers = []  # TODO create layers
    layers.append(FullyConnectedLayer(28 * 28, 128))
    layers.append(BatchNormalization(128))
    layers.append(ReLU())
    layers.append(FullyConnectedLayer(128, 32))
    layers.append(BatchNormalization(32))
    layers.append(ReLU())
    layers.append(FullyConnectedLayer(32, 10))
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


if __name__ == '__main__':
    main()
