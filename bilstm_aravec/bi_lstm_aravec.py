import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable

layers = keras.layers


@register_keras_serializable()
class BiLSTMAraVec(keras.Model):
    '''
    BiLSTM with BNG optimizer model, modified for word-level context embeddings using AraVec.
    Inputs:
        - char_ids: (batch_size, sequence_length)
        - word_embeddings: (batch_size, sequence_length, word_embedding_dim)
        - char_positions: (batch_size, sequence_length, 1)
    Outputs:
        - sigmoid binary_output: (batch_size, sequence_length, 1)
        - softmax class_output: (batch_size, sequence_length, 8)
    '''
    def __init__(
        self,
        input_vocab_size: int,
        word_embedding_dim: int,
        char_embedding_dim: int = 128,
        hidden_size: int = 250,
        binary_output_size: int = 1,
        class_output_size: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_vocab_size = input_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.hidden_size = hidden_size
        self.binary_output_size = binary_output_size
        self.class_output_size = class_output_size
        self.dropout_rate = dropout

        # trainable character embedding layer
        self.char_embedding = layers.Embedding(
            input_dim=input_vocab_size,
            output_dim=char_embedding_dim,
            mask_zero=False, 
            name="char_embedding",
        )

        # light dropout before recurrent layers
        self.dropout = layers.Dropout(dropout, name="input_dropout")

        # two stacked BiLSTM layers
        self.bi_lstm1 = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True),
            name="bi_lstm1",
        )
        self.bi_lstm2 = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True),
            name="bi_lstm2",
        )

        # output heads
        self.binary_output = layers.Dense(binary_output_size, activation="sigmoid", name="binary_output") # shadda output
        self.class_output = layers.Dense(class_output_size, activation="softmax", name="class_output") # other diacritics output

    def call(self, inputs, training=None):
        char_ids = inputs["char_ids"] # (batch_size, sequence_length)
        word_embeddings = inputs["word_embeddings"] # (batch_size, sequence_length, word_embedding_dim)
        char_positions = inputs["char_positions"] # (batch_size, sequence_length, 1)

        char_embeddings = self.char_embedding(char_ids) # (batch_size, sequence_length, char_embedding_dim)

        # concatenating all features resulting in (batch_size, sequence_length, char_embedding_dim + word_embedding_dim + 1)
        total_embeddings = keras.layers.concatenate([char_embeddings, word_embeddings, char_positions], axis=-1)

        total_embeddings = self.dropout(total_embeddings, training=training)

        total_embeddings = self.bi_lstm1(total_embeddings, training=training)
        total_embeddings = self.bi_lstm2(total_embeddings, training=training)

        binary_output = self.binary_output(total_embeddings)
        class_output = self.class_output(total_embeddings)

        return {
            "binary_output": binary_output,
            "class_output": class_output,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_vocab_size": self.input_vocab_size,
                "word_embedding_dim": self.word_embedding_dim,
                "char_embedding_dim": self.char_embedding_dim,
                "hidden_size": self.hidden_size,
                "binary_output_size": self.binary_output_size,
                "class_output_size": self.class_output_size,
                "dropout": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class BNGAdam(keras.optimizers.Adam):
    '''
    Block Normalized Gradient Adam Optimizer
    '''

    def __init__(self, block_size=64, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self._custom_block_size = block_size

    def _normalize_gradient_blocks(self, gradient):
        if gradient is None:
            return None

        original_shape = tf.shape(gradient)
        flat_grad = tf.reshape(gradient, [-1])
        total_size = tf.shape(flat_grad)[0]

        # calculate number of blocks
        num_blocks = tf.cast(
            tf.math.ceil(tf.cast(total_size, tf.float32) / tf.cast(self.block_size, tf.float32)),
            tf.int32,
        )

        # padding gradient if necessary
        pad_size = num_blocks * self.block_size - total_size
        padded_grad = tf.cond(
            pad_size > 0,
            lambda: tf.pad(flat_grad, [[0, pad_size]]),
            lambda: flat_grad,
        )

        # reshape into blocks
        blocks = tf.reshape(padded_grad, [num_blocks, self.block_size])

        # compute L2 norms of each block
        block_norms = tf.norm(blocks, axis=1, keepdims=True)

        # avoid division by zero
        block_norms = tf.maximum(block_norms, 1e-8)

        # normalise each block
        normalized_blocks = blocks / block_norms

        # reshape back to flat
        normalized_flat = tf.reshape(normalized_blocks, [-1])

        # remove padding if added
        normalized_flat = tf.cond(
            pad_size > 0,
            lambda: normalized_flat[:total_size],
            lambda: normalized_flat,
        )

        # reshape back to original shape
        normalized_grad = tf.reshape(normalized_flat, original_shape)

        return normalized_grad

    def apply_gradients(self, grads_and_vars, **kwargs):
        '''
        Apply block normalization to gradients before applying them
        Args:
            grads_and_vars: List of (gradient, variable) pairs.
        Returns:
            An `Operation` that applies the specified gradients.
        '''
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
