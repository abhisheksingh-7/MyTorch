import numpy as np


class Identity:
    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self):
        dAdZ = np.ones(self.A.shape, dtype="f")
        return dAdZ


class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        assert self.A.shape == Z.shape
        return self.A

    def backward(self):
        dAdZ = self.A - (self.A * self.A)
        assert dAdZ.shape == self.A.shape
        return dAdZ


class Tanh:
    def forward(self, Z):
        exp_Z = np.exp(Z)
        exp_negZ = np.exp(-Z)
        self.A = (exp_Z - exp_negZ) / (exp_Z + exp_negZ)
        assert self.A.shape == Z.shape
        return self.A

    def backward(self):
        dAdZ = 1 - (self.A * self.A)
        assert dAdZ.shape == self.A.shape
        return dAdZ


class ReLU:
    def forward(self, Z):
        self.A = np.maximum(np.zeros(Z.shape), Z)
        assert self.A.shape == Z.shape
        return self.A

    def backward(self):
        dAdZ = np.where(self.A > 0, 1, 0)
        assert dAdZ.shape == self.A.shape
        return dAdZ
