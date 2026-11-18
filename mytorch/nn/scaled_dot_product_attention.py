import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)  
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        self.Q = Q
        self.K = K
        self.V = V
        d_k = Q.shape[-1]  # dimension of the key vectors
        
        # Calculate attention scores
        scaled_dot_product = (Q @ np.swapaxes(K, -2, -1)) / np.sqrt(d_k)  # shape (N, ..., H, L, S)
        
        # Apply mask before softmax if provided. A fully masked query row must
        # have zero attention probabilities and zero output, without NaNs.
        # Keep the stored softmax values consistent for backward as well.
        full_mask = None
        if mask is not None:
            mask = np.broadcast_to(np.asarray(mask).astype(bool), scaled_dot_product.shape)
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)
            full_mask = np.all(mask, axis=-1, keepdims=True)  # shape (N, ..., H, L, 1)

        
        # Compute attention scores: 
        attention_scores = self.softmax.forward(scaled_dot_product)  # shape (N, ..., H, L, S)
        # # Think about which dimension you should apply Softmax
        if full_mask is not None:
            attention_scores = np.where(full_mask, 0.0, attention_scores)

            if hasattr(self, 'attention_scores'):
                self.softmax.A = attention_scores
        self.attention_scores = attention_scores

        # Calculate final output
        output = self.attention_scores @ V  # shape (N, ..., H, L, Ev)

        # Return final output
        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass
        A = self.attention_scores
        d_K = self.Q.shape[-1]  # dimension of the key vectors
    

        # Calculate gradients for V
        d_V = np.swapaxes(A, -1, -2) @ d_output  
        
        # Calculate gradients for attention scores
        d_attention_scores = d_output @ np.swapaxes(self.V, -2, -1)  # shape (N, ..., H, L, S)
        d_scaled_dot_product = A * (d_attention_scores - np.sum(d_attention_scores * A, axis=-1, keepdims=True))  # shape (N, ..., H, L, S)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_K)
        
        # Calculate gradients for Q and K
        d_Q = d_scaled_dot_product @ self.K  # shape (N, ..., H, L, E)  
        d_K = np.swapaxes(d_scaled_dot_product, -1, -2) @ self.Q 

        # Return gradients for Q, K, V
        return d_Q, d_K, d_V
