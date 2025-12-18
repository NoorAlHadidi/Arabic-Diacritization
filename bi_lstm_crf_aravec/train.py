import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from utils.preprocess import load_dataset
from utils.dataset_builder import (
    process_lines_with_word_indices,
    letters_id,
    diacritics,
    diacritics_id,
)
from .bi_lstm_crf import BiLSTM_CRF
from utils.word_embeddings import ArabicWordEmbedder
from utils.dictionary_builder import build_word_dictionary, save_dictionary
from utils.config import *


PAD_LABEL = 15

def encode_dataset_with_aravec(lines, word_embedder, cache_file=None):

    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached data: {cache_file}")
        with open(cache_file, "rb") as f:
            d = pickle.load(f)
        return d["char_ids"], d["word_embs"], d["targets"]

    print("Encoding dataset with AraVec...")

    all_letters, _, char_ids, targets, word_indices = \
        process_lines_with_word_indices(lines, letters_id, diacritics, diacritics_id)

    print("Computing word embeddings...")
    word_embs = word_embedder.get_batch_embeddings(all_letters, word_indices)

    char_ids = [np.array(x, np.int32) for x in char_ids]
    targets = [np.array(y, np.int32) for y in targets]
    word_embs = [w.astype(np.float32) for w in word_embs]

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "char_ids": char_ids,
                    "word_embs": word_embs,
                    "targets": targets,
                },
                f,
            )

    return char_ids, word_embs, targets

# CRF METRIC
def crf_viterbi_accuracy(y_true, y_pred_logits):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, PAD_LABEL)
    mask_f = tf.cast(mask, tf.float32)

    def decode_fn(logits, mask_np):
        preds = model.decode_batch(logits, mask_np)
        return preds.astype("int32")

    predictions = tf.py_function(
        decode_fn,
        [y_pred_logits, mask],
        tf.int32,
    )
    predictions.set_shape(y_true.shape)

    matches = tf.cast(tf.equal(predictions, y_true), tf.float32) * mask_f
    return tf.reduce_sum(matches) / tf.reduce_sum(mask_f)

if __name__ == "__main__":

    print("Starting training...")
    print("GPU:", tf.config.list_physical_devices("GPU"))

    ARAVEC_MDL_PATH = "/kaggle/input/aravec-100/full_uni_cbow_100_wiki.mdl"

    word_embedder = ArabicWordEmbedder(
        model_path=ARAVEC_MDL_PATH,
        embedding_dim=100,
    )

    output_dir = WORKING + "checkpoints"
    cache_dir = WORKING + "cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    train_lines = load_dataset(DATASET + "train.txt")
    val_lines = load_dataset(DATASET + "val.txt")
    all_lines = train_lines + val_lines


    print("Building and saving word dictionary...")
    word_dict = build_word_dictionary(all_lines)
    save_dictionary(word_dict, cache_dir + "/word_dictionary.pkl")


    char_ids, word_embs, targets = encode_dataset_with_aravec(
        all_lines,
        word_embedder,
        cache_file=os.path.join(cache_dir, "train_with_aravec.pkl"),
    )

    (
        train_char,
        val_char,
        train_word,
        val_word,
        train_tgt,
        val_tgt,
    ) = train_test_split(
        char_ids, word_embs, targets,
        test_size=0.15,
        random_state=42,
    )

    def train_gen():
        for c, w, y in zip(train_char, train_word, train_tgt):
            yield (c, w), y

    def val_gen():
        for c, w, y in zip(val_char, val_word, val_tgt):
            yield (c, w), y

    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            (
                tf.TensorSpec((None,), tf.int32),
                tf.TensorSpec((None, word_embedder.embedding_dim), tf.float32),
            ),
            tf.TensorSpec((None,), tf.int32),
        ),
    )

    val_ds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            (
                tf.TensorSpec((None,), tf.int32),
                tf.TensorSpec((None, word_embedder.embedding_dim), tf.float32),
            ),
            tf.TensorSpec((None,), tf.int32),
        ),
    )

    input_pad = letters_id["<PAD>"]

    train_ds = (
        train_ds
        .shuffle(10000)
        .bucket_by_sequence_length(
            lambda x, y: tf.shape(x[0])[0],
            [16, 32, 64, 96, 128],
            [256, 128, 96, 64, 32, 16],
            padded_shapes=(
                ([None], [None, word_embedder.embedding_dim]),
                [None],
            ),
            padding_values=((input_pad, 0.0), PAD_LABEL),
        )
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .bucket_by_sequence_length(
            lambda x, y: tf.shape(x[0])[0],
            [16, 32, 64, 96, 128],
            [256, 128, 96, 64, 32, 16],
            padded_shapes=(
                ([None], [None, word_embedder.embedding_dim]),
                [None],
            ),
            padding_values=((input_pad, 0.0), PAD_LABEL),
        )
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = BiLSTM_CRF(
        input_vocab_size=len(letters_id),
        word_embedding_dim=word_embedder.embedding_dim,
        num_tags=15,
        pad_label=PAD_LABEL,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(2e-3, clipnorm=1.0),
        loss=model.crf_loss,
        metrics=[crf_viterbi_accuracy],
    )

    steps_per_epoch = len(train_char) // 64
    val_steps = len(val_char) // 64

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, "best_model.keras"),
                monitor="val_crf_viterbi_accuracy",
                save_best_only=True,
                mode="max",
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
        ],
    )

    model.save(os.path.join(output_dir, "final_model.keras"))
    print("Training complete.")
