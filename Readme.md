Task 2: Implementing a Residual Block and ResNet-like Model

Description

This task involves implementing a Residual Block and integrating it into a simple ResNet-like model.

Steps Implemented

Residual Block Function:

Takes an input tensor.

Applies two Conv2D layers (each with 64 filters, kernel size (3x3), activation ReLU).

Adds a skip connection to sum the input tensor with the output before activation.

ResNet-like Model:

Initial Conv2D Layer with 64 filters, kernel size (7x7), stride 2, activation ReLU.

Two Residual Blocks applied sequentially.

Flatten Layer to convert feature maps into a dense layer input.

Dense Layer with 128 neurons, activation ReLU.

Output Layer with 10 neurons, activation Softmax.

Execution

Run the provided Python script to build and summarize the ResNet-like model.

The model summary is printed, showing the number of parameters and layers.
