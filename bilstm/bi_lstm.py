from keras.saving import register_keras_serializable
from tensorflow import keras
import tensorflow as tf
layers = keras.layers

@register_keras_serializable()
class BiLSTM(keras.Model):

    def __init__(self,
                 input_vocab_size,
                 embedding_dim=128,
                 hidden_size=250,
                 binary_output_size=1,
                 class_output_size=8,
                 **kwargs):

        super().__init__(**kwargs)

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.binary_output_size = binary_output_size
        self.class_output_size = class_output_size

        # Shared layers
        self.embedding = layers.Embedding(input_dim=input_vocab_size,
                                          output_dim=embedding_dim)

        self.bi_lstm1 = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True)
        )

        self.bi_lstm2 = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True)
        )

        # Output A (binary): shadda or not
        self.binary_output = layers.Dense(binary_output_size, activation="sigmoid")

        # Output B (8-class)
        self.class_output = layers.Dense(class_output_size, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        x = self.bi_lstm1(x)
        x = self.bi_lstm2(x)

        out_binary = self.binary_output(x)
        out_class = self.class_output(x)

        return {
            "binary_output": out_binary,
            "class_output": out_class
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_vocab_size": self.input_vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "binary_output_size": self.binary_output_size,
            "class_output_size": self.class_output_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# BNG Optimizer - use this instead of modifying the layer
@register_keras_serializable()
class BNGAdam(keras.optimizers.Adam):
    """Adam optimizer with Block Normalized Gradients"""
    
    def __init__(self, block_size=64, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self._custom_block_size = block_size
    
    def _normalize_gradient_blocks(self, gradient):
        """Normalize gradient in blocks"""
        if gradient is None:
            return None
            
        original_shape = tf.shape(gradient)
        flat_grad = tf.reshape(gradient, [-1])
        total_size = tf.shape(flat_grad)[0]
        
        # Calculate number of blocks
        num_blocks = tf.cast(tf.math.ceil(tf.cast(total_size, tf.float32) / 
                                          tf.cast(self.block_size, tf.float32)), tf.int32)
        
        # Pad gradient if necessary
        pad_size = num_blocks * self.block_size - total_size
        padded_grad = tf.cond(
            pad_size > 0,
            lambda: tf.pad(flat_grad, [[0, pad_size]]),
            lambda: flat_grad
        )
        
        # Reshape into blocks
        blocks = tf.reshape(padded_grad, [num_blocks, self.block_size])
        
        # Normalize each block
        block_norms = tf.sqrt(tf.reduce_sum(tf.square(blocks), axis=1, keepdims=True) + 1e-8)
        normalized_blocks = blocks / block_norms
        
        # Flatten back
        normalized_flat = tf.reshape(normalized_blocks, [-1])
        
        # Remove padding
        normalized_flat = tf.cond(
            pad_size > 0,
            lambda: normalized_flat[:total_size],
            lambda: normalized_flat
        )
        
        # Reshape to original shape
        normalized_grad = tf.reshape(normalized_flat, original_shape)
        
        return normalized_grad
    
    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply block normalization to gradients before applying them"""
        normalized_grads_and_vars = []
        
        for grad, var in grads_and_vars:
            if grad is not None:
                normalized_grad = self._normalize_gradient_blocks(grad)
                normalized_grads_and_vars.append((normalized_grad, var))
            else:
                normalized_grads_and_vars.append((grad, var))
        
        return super().apply_gradients(normalized_grads_and_vars, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self._custom_block_size})
        return config