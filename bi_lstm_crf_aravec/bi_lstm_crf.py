import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable

layers = keras.layers


@register_keras_serializable()
class BiLSTM_CRF(keras.Model):
    def __init__(
        self,
        input_vocab_size,
        word_embedding_dim,
        embedding_dim=128,
        hidden_size=250,
        num_tags=15,
        pad_label=15,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_vocab_size = input_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.pad_label = pad_label

        self.char_embedding = layers.Embedding(
            input_vocab_size, embedding_dim, name="char_embedding"
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
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="crf_transitions",
        )

    def call(self, inputs, training=None):
        """
        inputs: tuple/list -> (char_ids, word_embs)
          char_ids:   (B, T) int32
          word_embs:  (B, T, W) float32
        returns:
          logits:     (B, T, num_tags)
        """
        char_ids, word_embs = inputs  

        x_char = self.char_embedding(char_ids)          
        x = tf.concat([x_char, word_embs], axis=-1)     

        x = self.bi1(x, training=training)
        x = self.bi2(x, training=training)
        logits = self.dense(x)
        return logits

    # CRF helpers
    def _log_norm(self, logits, mask):
        logits = tf.cast(logits, tf.float32)
        mask_f = tf.cast(mask, tf.float32)

        max_len = tf.shape(logits)[1]
        alpha = logits[:, 0, :]  

        def time_step(t, alpha):
            emit_t = logits[:, t, :]  
            mask_t = tf.expand_dims(mask_f[:, t], 1)  

            alpha_exp = tf.expand_dims(alpha, 2)              
            trans_exp = tf.expand_dims(self.transitions, 0)   
            emit_exp = tf.expand_dims(emit_t, 1)              

            scores = alpha_exp + trans_exp + emit_exp         
            new_alpha = tf.reduce_logsumexp(scores, axis=1)   

            alpha = new_alpha * mask_t + alpha * (1.0 - mask_t)
            return t + 1, alpha

        t0 = tf.constant(1)
        cond = lambda t, *_: tf.less(t, max_len)
        _, alpha = tf.while_loop(
            cond,
            time_step,
            loop_vars=[t0, alpha],
            parallel_iterations=32,
        )

        return tf.reduce_logsumexp(alpha, axis=1)

    def _sequence_score(self, logits, labels, mask):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)
        mask_f = tf.cast(mask, tf.float32)

        batch_size = tf.shape(logits)[0]
        max_len = tf.shape(logits)[1]

        safe_labels = tf.where(mask, labels, tf.zeros_like(labels))

        batch_indices = tf.range(batch_size)[:, None]
        time_indices = tf.range(max_len)[None, :]
        batch_time = tf.stack(
            [
                tf.tile(batch_indices, [1, max_len]),
                tf.tile(time_indices, [batch_size, 1]),
                safe_labels,
            ],
            axis=-1,
        )

        emissions = tf.gather_nd(logits, batch_time)  # (B, T)
        emission_score = tf.reduce_sum(emissions * mask_f, axis=1)

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
            no_transition_score,
        )

        return emission_score + transition_score

    def neg_log_likelihood(self, y_true, logits):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.pad_label)

        logZ = self._log_norm(logits, mask)
        path_score = self._sequence_score(logits, y_true, mask)

        nll = logZ - path_score
        nll = tf.clip_by_value(nll, 0.0, 1e6)
        return nll

    def crf_loss(self, y_true, y_pred):
        return tf.reduce_mean(self.neg_log_likelihood(y_true, y_pred))

    def viterbi_decode(self, emissions, seq_len=None):
        import numpy as np

        if isinstance(emissions, tf.Tensor):
            emissions = emissions.numpy()
        trans = self.transitions.numpy()

        T, C = emissions.shape
        if seq_len is None:
            seq_len = T
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

        for t in range(actual_T, T):
            tags[t] = 0

        return tags

    def decode_batch(self, logits, mask=None):
        import numpy as np

        logits_np = logits.numpy() if isinstance(logits, tf.Tensor) else logits
        B, T, _ = logits_np.shape

        predictions = np.zeros((B, T), dtype=np.int32)

        for i in range(B):
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
        cfg.update(
            {
                "input_vocab_size": self.input_vocab_size,
                "word_embedding_dim": self.word_embedding_dim,  # ✅ important
                "embedding_dim": self.embedding_dim,
                "hidden_size": self.hidden_size,
                "num_tags": self.num_tags,
                "pad_label": self.pad_label,
            }
        )
        return cfg

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("dtype", None)
        config.pop("name", None)
        config.pop("trainable", None)
        return cls(**config)
