# Convolutional Similarity Regularization
A repo contraining all the code and scripts for the paper proposing the Convolutional Similarity regularization.

The Convolutional Similarity regularization is a method to minimize convolutional feature maps redundancy by reducing their cosine similarity through kernel optimization.

# CUDA Library Install
The Convolutional Similarity regularization is only available in a GPU-compatible version. The directory lib has the setup files and the library itself. Open the `lib\` directory and run the following command to install

`pip install .`

Tested on
`PyTorch 2.0.1`
`CUDA 11.8`
`Python 3.10.11`

**This step is necessary to be able to use Convolutional Similarity regularization.**

# Using Convolutional Similarity Regularization

It can be incorporated into any model containing 2D Convolutional layer by simly adding the following term to the loss

`loss_conv_sim = ConvSim2DLoss(model)`

The function `ConvSim2DLoss()` is defined in `ConvSimFunctions.py`.

