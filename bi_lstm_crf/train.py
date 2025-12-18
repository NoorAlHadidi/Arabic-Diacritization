import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from utils.preprocess import load_dataset
from utils.dataset_builder import process_lines, letters_id, diacritics, diacritics_id
from .bi_lstm_crf import BiLSTM_CRF
from utils.config import *

PAD_LABEL = 15  


def encode_dataset(lines, cache_file=None):
    """
    Encode dataset and optionally load/save cache.
    Returns (inputs, targets) as numpy arrays of Python lists.
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached data: {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["inputs"], data["targets"]

    print("Encoding dataset...")

    _, _, inputs, targets = process_lines(
        lines, letters_id, diacritics, diacritics_id
    )

    inputs = np.array(inputs, dtype=object)
    targets = np.array(targets, dtype=object)

    if cache_file:
        print(f"Saving encoded data to: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"inputs": inputs, "targets": targets}, f)

    return inputs, targets


# CRF METRIC: Viterbi decoding accuracy

def crf_viterbi_accuracy(y_true, y_pred_logits):
    """
    y_true: (B, T)
    y_pred_logits: (B, T, C)

    Computes accuracy using REAL CRF decoding (Viterbi),
    ignoring padded positions (label == PAD_LABEL).
    """
    y_true = tf.cast(y_true, tf.int32)

    mask = tf.not_equal(y_true, PAD_LABEL)        
    mask_f = tf.cast(mask, tf.float32)

    def decode_fn(logits, mask_np):
        preds = model.decode_batch(logits, mask_np)
        return preds.astype("int32")

    predictions = tf.py_function(
        func=decode_fn,
        inp=[y_pred_logits, mask],
        Tout=tf.int32,
    )

    predictions.set_shape(y_true.shape)  

    matches = tf.cast(tf.equal(predictions, y_true), tf.float32) * mask_f
    correct = tf.reduce_sum(matches)
    total = tf.reduce_sum(mask_f)

    return tf.math.divide_no_nan(correct, total)


if __name__ == "__main__":
    print("Starting training pipeline...")
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    output_dir = WORKING + "checkpoints"
    cache_dir = WORKING + "cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    train_file = DATASET + "train.txt"
    val_file = DATASET + "val.txt"

    print("\nLoading datasets...")
    train_lines = load_dataset(train_file)
    val_lines = load_dataset(val_file)

    big_train = train_lines + val_lines
    print(f"Loaded {len(big_train)} total training lines")

    train_inputs, train_targets = encode_dataset(
        big_train,
        os.path.join(cache_dir, "train_encoded.pkl")
    )

    print("\nDataset sizes:")
    print(f"  Train sequences: {len(train_inputs)}")

    lengths = [len(seq) for seq in train_inputs]
    print("\nSequence length statistics:")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Mean:   {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")

    # Split into train/validation
    print("\nSplitting train/validation...")
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        train_inputs, train_targets,
        test_size=0.15, random_state=42
    )
    print(f"  Train: {len(train_inputs)} sequences")
    print(f"  Val:   {len(val_inputs)} sequences")

    def train_gen():
        for x, y in zip(train_inputs, train_targets):
            yield np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def val_gen():
        for x, y in zip(val_inputs, val_targets):
            yield np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    # TF.DATA DATASETS WITH BUCKETING
    print("\nBuilding TensorFlow datasets...")

    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None,), tf.int32),
        ),
    )

    val_ds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None,), tf.int32),
        ),
    )

    input_pad = letters_id["<PAD>"]
    target_pad = PAD_LABEL

    bucket_boundaries = [16, 32, 64, 96, 128, 156, 212, 256]
    bucket_batch_sizes = [256, 128, 128, 96, 96, 64, 64, 64, 32]

    print("\nBucket configuration:")
    print("  Boundaries:", bucket_boundaries)
    print("  Batch sizes:", bucket_batch_sizes)

    train_ds = train_ds.shuffle(
        buffer_size=min(20000, len(train_inputs)),
        seed=42,
        reshuffle_each_iteration=True,
    )

    train_ds = train_ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=([None], [None]),
        padding_values=(input_pad, target_pad),
    )

    val_ds = val_ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[min(b * 2, 256) for b in bucket_batch_sizes],
        padded_shapes=([None], [None]),
        padding_values=(input_pad, target_pad),
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    print("Datasets ready")

    # Build model
    model = BiLSTM_CRF(
        input_vocab_size=len(letters_id),
        num_tags=15,
        pad_label=PAD_LABEL,
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=2e-3,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=model.crf_loss,
        metrics=[crf_viterbi_accuracy],
    )

    print("Model compiled successfully!")

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.keras"),
            monitor="val_crf_viterbi_accuracy",  
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"),
            append=True,
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "epoch_{epoch:02d}.keras"),
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        ),
    ]

    # Train
    print("\nStarting training...")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {history_path}")

    print("\nSaved files:")
    for file in sorted(os.listdir(output_dir)):
        p = os.path.join(output_dir, file)
        size = os.path.getsize(p) / (1024 * 1024)
        print(f"  - {file} ({size:.2f} MB)")

    print("\nTraining complete!")
