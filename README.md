# Deep Generative Deconvolutional Network (DGDN)

This repository contains a PyTorch implementation of the DGDN model, which is described in [Variational Autoencoder for Deep Learning of Images, Labels and Captions](https://proceedings.neurips.cc/paper/2016/file/eb86d510361fc23b59f18c1bc9802cc6-Paper.pdf).

The repository contains two files:

* `model.py` - contains the implementation of the model itself. It has 3 classes:
    * `Pool` - implements stochastic pooling
    * `Unpool` - implements stochastic unpooling
    * `DGDN` - implements the actual DGDN model
* `train.py` - has the training loop to train the model. It has 5 functions:
  * `loss_func` - implements the loss function as described in the paper mentioned above.
  * `prepare_dataset` - downloads the MNIST dataset to `data/mnist` and returns PyTorch data loaders for training and testing.
  * `train` - contains the training loop.
  * `test` - evaluates the value of the loss function on the test dataset.
  * `main` - acts as the entrypoint for the program.