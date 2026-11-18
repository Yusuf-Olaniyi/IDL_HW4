import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        
        Z = A @ self.W.T + self.b

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        in_features = self.A.shape[-1]
        out_features = dLdZ.shape[-1]
        A_2d = self.A.reshape(-1, in_features)        # (batch, in_features)
        dLdZ_2d = dLdZ.reshape(-1, out_features)      # (batch, out_features)

        # Compute gradients
        self.dLdA = dLdZ_2d @ self.W                  # (batch, in_features)
        self.dLdW = dLdZ_2d.T @ A_2d                  # (out_features, in_features)
        self.dLdb = np.sum(dLdZ_2d, axis=0)           # (out_features,)
        self.dLdA = self.dLdA.reshape(self.A.shape)   # (*, in_features)

        # Return gradient of loss wrt input
        return self.dLdA
