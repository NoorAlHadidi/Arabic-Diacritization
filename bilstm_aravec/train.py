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
from utils.dictionary_builder import build_word_dictionary, save_dictionary
from .bi_lstm_aravec import BiLSTMAraVec, BNGAdam
from utils.word_embeddings import ArabicWordEmbedder
from utils.config import *

def extract_binary_label(target_labels):
    '''
    - if diacritic has shadda (7-13) -> 1
    - else -> 0
    Returns: (sequence_length, 1)
    '''
    shadda_set = {7, 8, 9, 10, 11, 12, 13}
    return np.array([[1 if label in shadda_set else 0] for label in target_labels], dtype=np.int32)


def extract_class_label(target_labels):
    '''
    - base diacritics (0-6) remain unchanged
    - diacritics with shadda (8-13) are mapped to their base forms (0-5)
    - shadda-only (7) and no-diacritic (14) are mapped to class 7
    Returns: (sequence_length,) array of class labels in range [0-7]
    '''
    mapping = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7
    }
    return np.array([mapping[label] for label in target_labels], dtype=np.int32)


def compute_char_positions(letters, word_indices):
    '''
    Returns: (seq_len, 1) array of normalised character positions within their words
    '''
    positions = np.zeros((len(letters), 1), dtype=np.float32)

    words = {}  
    for char_index, word_index in enumerate(word_indices):
        if word_index == -1: # punctuation or space
            continue
        words.setdefault(word_index, []).append(char_index)

    for word_index, char_pos_list in words.items():
        length = len(char_pos_list)
        if length == 1:
            positions[char_pos_list[0]] = 0.5 # single-char word
        else:
            for index, position in enumerate(char_pos_list):
                positions[position] = index / (length - 1)

    return positions


def encode_dataset_with_word_embeddings(lines, word_embedder, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached data: {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["char_ids"], data["word_embeddings"], data["char_positions"], data["targets"]

    print("Encoding dataset with word embeddings.")

    all_letters, all_labels, char_ids, targets, word_indices = process_lines_with_word_indices(
        lines, letters_id, diacritics, diacritics_id
    )

    print("  Computing AraVec word embeddings (broadcast per character).")
    word_embeddings = word_embedder.get_batch_embeddings(all_letters, word_indices)

    print("  Computing character position features.")
    char_positions = [
        compute_char_positions(letters, words_index)
        for letters, words_index in zip(all_letters, word_indices)
    ]

    char_ids = [np.array(x, dtype=np.int32) for x in char_ids] # (seq_len,)
    targets = [np.array(y, dtype=np.int32) for y in targets] # (seq_len,)
    word_embeddings = [we.astype(np.float32) for we in word_embeddings] # (seq_len, embedding_dimension)
    char_positions = [cp.astype(np.float32) for cp in char_positions] # (seq_len, 1)

    if cache_file:
        print(f"Saving cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "char_ids": char_ids,
                    "word_embeddings": word_embeddings,
                    "char_positions": char_positions,
                    "targets": targets,
                },
                f,
            )

    return char_ids, word_embeddings, char_positions, targets


if __name__ == "__main__":
    print("=" * 50)
    print("Training BiLSTM with AraVec Word Context")
    print("=" * 50)

    print("\nGPU devices:", tf.config.list_physical_devices("GPU"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 50)
    print("Loading AraVec")
    print("=" * 50)

    word_embedder = ArabicWordEmbedder(
        model_path=ARAVEC_MODEL_PATH,
        embedding_dim=100,  
    )

    print("\n" + "=" * 50)
    print("Loading Dataset")
    print("=" * 50)

    train_lines = load_dataset(TRAIN_DATASET_PATH)
    val_lines = load_dataset(VAL_DATASET_PATH)

    big_train = train_lines + val_lines
    print(f"Loaded total lines: {len(big_train)}")

    word_dictionary = build_word_dictionary(big_train)
    save_dictionary(word_dictionary, DICTIONARY_PATH)

    char_ids, word_embs, char_pos, targets = encode_dataset_with_word_embeddings(
    big_train,
    word_embedder,
    cache_file=None, 
    )

    lengths = [len(x) for x in char_ids]
    print(f"\nSequence length statistics: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")

    # split into train/val
    (train_char_ids, val_char_ids,
     train_word_emb, val_word_emb,
     train_pos, val_pos,
     train_targets, val_targets) = train_test_split(
        char_ids, word_embs, char_pos, targets,
        test_size=0.05, random_state=42
    )

    print(f"\nTrain sequences: {len(train_char_ids)} | Validation sequences: {len(val_char_ids)}")

    # data generators
    def train_generate():
        for char, word, position, target in zip(train_char_ids, train_word_emb, train_pos, train_targets):
            yield (
                {"char_ids": char, "word_embeddings": word, "char_positions": position},
                {"binary_output": extract_binary_label(target), "class_output": extract_class_label(target)},
            )

    def val_generate():
        for char, word, position, target in zip(val_char_ids, val_word_emb, val_pos, val_targets):
            yield (
                {"char_ids": char, "word_embeddings": word, "char_positions": position},
                {"binary_output": extract_binary_label(target), "class_output": extract_class_label(target)},
            )

    print("\nBuilding tf.data datasets...")

    train_dataset = tf.data.Dataset.from_generator(
        train_generate,
        output_signature=(
            {
                "char_ids": tf.TensorSpec((None,), tf.int32),
                "word_embeddings": tf.TensorSpec((None, word_embedder.embedding_dim), tf.float32),
                "char_positions": tf.TensorSpec((None, 1), tf.float32),
            },
            {
                "binary_output": tf.TensorSpec((None, 1), tf.int32),
                "class_output": tf.TensorSpec((None,), tf.int32),
            },
        ),
    )

    val_dataset = tf.data.Dataset.from_generator(
        val_generate,
        output_signature=(
            {
                "char_ids": tf.TensorSpec((None,), tf.int32),
                "word_embeddings": tf.TensorSpec((None, word_embedder.embedding_dim), tf.float32),
                "char_positions": tf.TensorSpec((None, 1), tf.float32),
            },
            {
                "binary_output": tf.TensorSpec((None, 1), tf.int32),
                "class_output": tf.TensorSpec((None,), tf.int32),
            },
        ),
    )

    input_pad = letters_id["<PAD>"]
    binary_pad = 0 # no-shadda
    class_pad = 7 # no-diacritic

    # bucketing
    bucket_boundaries = [16, 32, 48, 64, 80]
    bucket_batch_sizes = [64, 48, 32, 24, 16, 12]

    train_dataset = train_dataset.shuffle(buffer_size=5000, seed=42, reshuffle_each_iteration=True)

    train_dataset = train_dataset.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x["char_ids"])[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=(
            {
                "char_ids": [None],
                "word_embeddings": [None, word_embedder.embedding_dim],
                "char_positions": [None, 1],
            },
            {
                "binary_output": [None, 1],
                "class_output": [None],
            },
        ),
        padding_values=(
            {
                "char_ids": input_pad,
                "word_embeddings": 0.0,
                "char_positions": 0.0,
            },
            {
                "binary_output": binary_pad,
                "class_output": class_pad,
            },
        ),
    )

    val_dataset = val_dataset.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x["char_ids"])[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[min(b * 2, 256) for b in bucket_batch_sizes],
        padded_shapes=(
            {
                "char_ids": [None],
                "word_embeddings": [None, word_embedder.embedding_dim],
                "char_positions": [None, 1],
            },
            {
                "binary_output": [None, 1],
                "class_output": [None],
            },
        ),
        padding_values=(
            {
                "char_ids": input_pad,
                "word_embeddings": 0.0,
                "char_positions": 0.0,
            },
            {
                "binary_output": binary_pad,
                "class_output": class_pad,
            },
        ),
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    print("Train & Val Datasets are ready.")

    print("\nBuilding Model.")

    model = BiLSTMAraVec(
        input_vocab_size=len(letters_id),
        word_embedding_dim=word_embedder.embedding_dim,
        char_embedding_dim=128,
        hidden_size=250,
        binary_output_size=1,
        class_output_size=8,
        dropout=0.2,
    )


    optimizer = BNGAdam(
        learning_rate=1e-3,
        block_size=64,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "binary_output": keras.losses.BinaryCrossentropy(),
            "class_output": keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "binary_output": ["accuracy"],
            "class_output": ["accuracy"],
        },
    )

    for batch in train_dataset.take(1):
        model(batch[0])
        model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_class_output_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, "training_log.csv"),
            append=True,
        ),
    ]

    print("\nStarting Training.")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(OUTPUT_DIR, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    with open(os.path.join(OUTPUT_DIR, "training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    print("\nTraining Complete.")
