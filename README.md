
⚠️ **IMPORTANT DISCLAIMER**: This repository contains code from an academic assignment completed at Carnegie Mellon University (CMU). It is shared here solely to showcase my work. **Please do not use, reproduce, distribute, or reference this code in any form**. The intention is purely to display the project and not to provide a solution or reference for current or future students. If you're a student at CMU or any other institution, please adhere to your university's academic integrity guidelines. Misuse of this code is a violation of the repository's license.

---

# MyTorch®

This repository is an implementation of a deep learning library written in Python. Inspired by [PyTorch](https://pytorch.org/), this library, `MyTorch®` is used to create everything from multilayer perceptrons (MLP), convolutional neural networks (CNN), to recurrent neural networks with gated recurrent units (GRU) and long-short term memory (LSTM) structures.

`MyTorch®` is structurally similar to to [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) and its modules can be reused in other applications.

## Setup
1. Clone the repository and run `poetry install`.
2. Run `make help` for further instructions.

## Reference
- MLP:
    - Linear Layer `[mytorch.nn.Linear]`
    - Activation Functions:
        - Sigmoid `[mytorch.nn.Sigmoid]`
        - ReLU `[mytorch.nn.ReLU]`
        - Tanh `[mytorch.nn.Tanh]`
    - Neural Network Models:
        - MLP0 (Hidden Layers = 0) `[mytorch.models.MLP0]`
        - MLP1 (Hidden Layers = 1) `[mytorch.models.MLP1]`
        - MLP4 (Hidden Layers = 4) `[mytorch.models.MLP4]`
    - Criterion: Loss Functions
        - MSE Loss `[mytorch.nn.MSELoss]`
        - Cross-Entropy Loss `[mytorch.nn.CrossEntropyLoss]`
    - Optimizers:
        - Stochastic Gradient Descent (SGD) `[mytorch.optim.SGD]`
    - Regularization:
        - Batch Normalization `[mytorch.nn.BatchNorm1d]`
- CNN:
    - Resampling:
        - Upsample1d `[mytorch.nn.Upsample1d]`
        - Downsampling1d `[mytorch.nn.Downsampling1d]`
        - Upsampling2d `[mytorch.nn.Upsampling2d]`
        - Downsampling2d `[mytorch.nn.Downsampling2d]`
    - Convolutional Layer:
        - Conv1d_stride1 `[mytorch.nn.Conv1d_stride1]`
        - Conv1d `[mytorch.nn.Conv1d]`
        - Conv2d_stride1 `[mytorch.nn.Conv2d_stride1]`
        - Conv2d `[mytorch.nn.Conv2d]`
    - Transposed Convolution:
        - ConvTranspose1d `[mytorch.nn.ConvTranspose1d]`
        - ConvTranspose2d `[mytorch.nn.ConvTranspose2d]`
    - Pooling:
        - MaxPool2d_stride1 or MeanPool2d_stride1 `[mytorch.nn.MaxPool2d_stride1]` or `[mytorch.nn.MeanPool2d_stride1]` 
        - MaxPool2d or MeanPool2d `[mytorch.nn.MaxPool2d]` or `[mytorch.nn.MeanPool2d]`
- RNN:
    - RNN Cell `[mytorch.nn.rnn_cell]`
    - RNN Phoneme Classifier `[models.rnn_classifier]`
    - GRU Cell `[mytorch.nn.gru_cell]`
    - CTC `[mytorch.nn.ctc]`
        - CTC Loss
    - CTC Decoding: Greedy Search and Beam Search `[mytorch.nn.CTCDecoding]`






---
Made with :coffee: as part of Carnegie Mellon University's [11-785 Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S23/index.html).

