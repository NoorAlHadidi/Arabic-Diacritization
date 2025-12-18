import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable

layers = keras.layers


@register_keras_serializable()
class BiLSTM_CRF(keras.Model):
    def __init__(
        self,
        input_vocab_size,
        embedding_dim=128,
        hidden_size=250,
        num_tags=15,
        pad_label=15,
        **kwargs
    ):
        """
        BiLSTM + CRF implemented in pure TensorFlow.

        Args:
            input_vocab_size: size of letter2idx vocab.
            embedding_dim: embedding size.
            hidden_size: LSTM hidden size (per direction).
            num_tags: number of real labels (0..num_tags-1).
            pad_label: label id used ONLY for padding (e.g. 15).
        """
        super().__init__(**kwargs)

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.pad_label = pad_label

        self.embedding = layers.Embedding(
            input_vocab_size, embedding_dim, name="embedding"
        )
        self.bi1 = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True),
            name="bi_lstm_1",
        )
        self.bi2 = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True),
            name="bi_lstm_2",
        )
        self.dense = layers.Dense(num_tags, name="emission_dense")

        self.transitions = self.add_weight(
            shape=(num_tags, num_tags),
            initializer=keras.initializers.GlorotUniform(),  # Better than random_uniform
            trainable=True,
            name="crf_transitions",
        )

    def call(self, inputs, training=None):
        """
        inputs: (batch, seq_len) int32 of token ids.
        returns: (batch, seq_len, num_tags) emission scores (logits).
        """
        x = self.embedding(inputs)
        x = self.bi1(x, training=training)
        x = self.bi2(x, training=training)
        logits = self.dense(x)
        return logits

    # CRF helpers
    def _log_norm(self, logits, mask):
        """
        Compute log Z(x) with forward algorithm.

        logits: (B, T, C)
        mask:   (B, T) bool
        returns: (B,)
        """
        logits = tf.cast(logits, tf.float32)
        mask_f = tf.cast(mask, tf.float32)

        batch_size = tf.shape(logits)[0]
        max_len = tf.shape(logits)[1]

        # alpha_0 = emissions at t=0
        alpha = logits[:, 0, :]  # (B, C)

        # iterate over time
        def time_step(t, alpha):
            emit_t = logits[:, t, :]  # (B, C)
            mask_t = tf.expand_dims(mask_f[:, t], 1)  # (B, 1)

            # shape: (B, C, 1) + (1, C, C) + (B, 1, C) -> (B, C, C)
            alpha_exp = tf.expand_dims(alpha, 2)
            trans_exp = tf.expand_dims(self.transitions, 0)
            emit_exp = tf.expand_dims(emit_t, 1)

            scores = alpha_exp + trans_exp + emit_exp  # (B, C, C)
            new_alpha = tf.reduce_logsumexp(scores, axis=1)  # (B, C)

            # if mask_t == 0 → keep previous alpha
            alpha = new_alpha * mask_t + alpha * (1.0 - mask_t)
            return t + 1, alpha

        # loop from t=1..max_len-1
        t0 = tf.constant(1)
        cond = lambda t, *_: tf.less(t, max_len)
        _, alpha = tf.while_loop(
            cond,
            time_step,
            loop_vars=[t0, alpha],
            parallel_iterations=32,
        )

        logZ = tf.reduce_logsumexp(alpha, axis=1)  # (B,)
        return logZ

    def _sequence_score(self, logits, labels, mask):
        """
        Compute score for the given label sequence.

        logits: (B, T, C)
        labels: (B, T) int32
        mask:   (B, T) bool
        returns: (B,)
        """
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)
        mask_f = tf.cast(mask, tf.float32)

        batch_size = tf.shape(logits)[0]
        max_len = tf.shape(logits)[1]
        num_tags = self.num_tags

        # Safe labels: for padded positions, set to 0 (will be masked out)
        safe_labels = tf.where(mask, labels, tf.zeros_like(labels))

        # Emission scores
        batch_indices = tf.range(batch_size)[:, None]  
        time_indices = tf.range(max_len)[None, :]      
        batch_time = tf.stack([
            tf.tile(batch_indices, [1, max_len]),
            tf.tile(time_indices, [batch_size, 1]),
            safe_labels
        ], axis=-1) 

        emissions = tf.gather_nd(logits, batch_time)  
        emission_score = tf.reduce_sum(emissions * mask_f, axis=1)  

        # Transition scores (only compute if T > 1)
        # Use tf.cond to handle single-token sequences
        def compute_transition_score():
            labels_prev = safe_labels[:, :-1] 
            labels_curr = safe_labels[:, 1:]  

            trans_mask = mask[:, 1:] & mask[:, :-1]  
            trans_mask_f = tf.cast(trans_mask, tf.float32)

            pairs = tf.stack([labels_prev, labels_curr], axis=2)  
            trans_scores = tf.gather_nd(self.transitions, pairs)  

            return tf.reduce_sum(trans_scores * trans_mask_f, axis=1)  
        
        def no_transition_score():
            return tf.zeros([batch_size], dtype=tf.float32)
        
        transition_score = tf.cond(
            tf.greater(max_len, 1),
            compute_transition_score,
            no_transition_score
        )

        return emission_score + transition_score

    def neg_log_likelihood(self, y_true, logits):
        """
        y_true: (B, T) int32 (0..num_tags-1 or pad_label)
        logits: (B, T, C)
        """
        y_true = tf.cast(y_true, tf.int32)
        # mask out padding positions (label == pad_label)
        mask = tf.not_equal(y_true, self.pad_label)  # (B, T)

        logZ = self._log_norm(logits, mask)
        path_score = self._sequence_score(logits, y_true, mask)
        
        # add numerical stability
        nll = logZ - path_score
        # Clip to prevent extreme values
        nll = tf.clip_by_value(nll, 0.0, 1e6)
        
        return nll 

    def crf_loss(self, y_true, y_pred):
        """
        Keras-compatible loss(y_true, y_pred) → scalar
        y_pred is logits from call().
        """
        losses = self.neg_log_likelihood(y_true, y_pred)
        return tf.reduce_mean(losses)

    # Viterbi decode with mask handling
    def viterbi_decode(self, emissions, seq_len=None):
        """
        emissions: (T, C) tensor or np.array
        seq_len: actual sequence length (excluding padding), optional
        returns: 1D np.array of size T with best tag indices.
        """
        import numpy as np

        if isinstance(emissions, tf.Tensor):
            emissions = emissions.numpy()
        trans = self.transitions.numpy()

        T, C = emissions.shape
        
        # If seq_len not provided, use full length
        if seq_len is None:
            seq_len = T
        
        # Only decode up to actual sequence length
        actual_T = min(seq_len, T)
        
        dp = np.full((T, C), -1e10, dtype=np.float32)
        backp = np.zeros((T, C), dtype=np.int32)

        dp[0] = emissions[0]

        for t in range(1, actual_T):
            for j in range(C):
                scores = dp[t - 1] + trans[:, j]
                best_prev = np.argmax(scores)
                dp[t, j] = scores[best_prev] + emissions[t, j]
                backp[t, j] = best_prev

        best_last = int(np.argmax(dp[actual_T - 1]))
        tags = np.zeros(T, dtype=np.int32)
        tags[actual_T - 1] = best_last

        for t in range(actual_T - 2, -1, -1):
            tags[t] = int(backp[t + 1, tags[t + 1]])
        
        # For padding positions, use tag 0 (arbitrary, will be ignored)
        for t in range(actual_T, T):
            tags[t] = 0

        return tags

    # Batch Viterbi decode
    def decode_batch(self, logits, mask=None):
        """
        Decode a batch of sequences.
        
        logits: (B, T, C) emissions
        mask: (B, T) bool mask, optional
        returns: (B, T) predicted tags
        """
        import numpy as np
        
        logits_np = logits.numpy() if isinstance(logits, tf.Tensor) else logits
        B, T, C = logits_np.shape
        
        predictions = np.zeros((B, T), dtype=np.int32)
        
        for i in range(B):
            # Get sequence length from mask
            if mask is not None:
                mask_np = mask.numpy() if isinstance(mask, tf.Tensor) else mask
                seq_len = int(np.sum(mask_np[i]))
            else:
                seq_len = T
            
            predictions[i] = self.viterbi_decode(logits_np[i], seq_len)
        
        return predictions

    # Serialization
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "input_vocab_size": self.input_vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_tags": self.num_tags,
            "pad_label": self.pad_label,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('dtype', None)
        config.pop('name', None)
        config.pop('trainable', None)
        return cls(**config)