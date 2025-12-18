import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from utils.dictionary_builder import build_word_dictionary, save_dictionary
from utils.preprocess import load_dataset
from utils.dataset_builder import process_lines, letters_id, diacritics, diacritics_id
from .bi_lstm import BNGAdam, BiLSTM
from utils.config import *

# -----------------------------
#  MAPPING LOGIC
# -----------------------------

def extract_binary_label(tgt):
    """
    Return (seq_len, 1)
    1 = has shadda
    0 = no shadda
    """
    shadda_set = {7, 8, 9, 10, 11, 12, 13}
    return np.array([[1 if t in shadda_set else 0] for t in tgt], dtype=np.int32)


def extract_class_label(tgt):
    """
    Return (seq_len,)
    8 class mapping:
        0 fatha
        1 fathatan
        2 damma
        3 dammatan
        4 kasra
        5 kasratan
        6 sukun (includes empty, shadda-only)
        7 no diactric
    """
    mapping = {
        0: 0,  # fatha
        1: 1,  # fathatan
        2: 2,  # damma
        3: 3,  # dammatan
        4: 4,  # kasra
        5: 5,  # kasratan
        6: 6,  # sukun
        7: 7,  # shadda-only → EMPTY
        8: 0,  # shadda fatha
        9: 1,  # shadda fathatan
        10: 2, # shadda damma
        11: 3, # shadda dammatan
        12: 4, # shadda kasra
        13: 5, # shadda kasratan
        14: 7  # EMPTY → no diactric
    }

    return np.array([mapping[t] for t in tgt], dtype=np.int32)


# ------------------------------------------------
#  ENCODING THE DATASET
# ------------------------------------------------

def encode_dataset(lines, cache_file=None):
    """
    Encode dataset with optional caching.
    
    Returns:
        Tuple of (inputs, targets) as numpy arrays
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached data: {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['inputs'], data['targets']

    print("Encoding dataset...")
    
    # Process lines and get encoded data
    _, _, inputs, targets = process_lines(lines, letters_id, diacritics, diacritics_id)
    
    # Convert to numpy arrays
    inputs = np.array(inputs, dtype=object)
    targets = np.array(targets, dtype=object)

    # Cache the data
    if cache_file:
        print(f"Saving to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'inputs': inputs, 'targets': targets}, f)

    return inputs, targets


# ----------------------
#       MAIN
# ----------------------

if __name__ == "__main__":
    
    print("Starting training pipeline...")
    
    # # Enable mixed precision for better GPU utilization
    # print("Enabling mixed precision training...")
    # keras.mixed_precision.set_global_policy("mixed_float16")
    
    # Check GPU availability
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

    # Setup directories
    output_dir = WORKING
    cache_dir = WORKING + "cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    train_file = DATASET + "train.txt"
    test_file = DATASET + "val.txt"

    # Load raw lines
    print("\nLoading datasets...")
    train_lines = load_dataset(train_file)
    test_lines = load_dataset(test_file)
    all_lines = train_lines + test_lines
    print(f"Loaded {len(all_lines)} training lines")
    
    # Build dictionary
    word_dict = build_word_dictionary(all_lines)
    
    # Save dictionary
    dict_path = WORKING + "cache/word_dictionary.pkl"
    save_dictionary(word_dict, dict_path)

    # Encode datasets with caching
    train_inputs, train_targets = encode_dataset(
        all_lines,
        os.path.join(cache_dir, "train_encoded.pkl")
    )


    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_inputs)} sequences")
    
    # Print sequence length statistics
    train_lengths = [len(x) for x in train_inputs]
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(train_lengths)}")
    print(f"  Max: {max(train_lengths)}")
    print(f"  Mean: {np.mean(train_lengths):.2f}")
    print(f"  Median: {np.median(train_lengths):.2f}")

    # Split train/validation
    print("\nSplitting train/validation...")
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        train_inputs, train_targets, test_size=0.10, random_state=42
    )
    
    print(f"  Train: {len(train_inputs)} sequences")
    print(f"  Val: {len(val_inputs)} sequences")

    # --------------------
    # Dataset Generators
    # --------------------

    def train_gen():
        for x, y in zip(train_inputs, train_targets):
            yield np.array(x, dtype=np.int32), {
                "binary_output": extract_binary_label(y),
                "class_output": extract_class_label(y)
            }

    def val_gen():
        for x, y in zip(val_inputs, val_targets):
            yield np.array(x, dtype=np.int32), {
                "binary_output": extract_binary_label(y),
                "class_output": extract_class_label(y)
            }

    # --------------------
    # Build tf.data Datasets
    # --------------------

    print("\nBuilding TensorFlow datasets...")
    
    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec((None,), tf.int32),
            {
                "binary_output": tf.TensorSpec((None, 1), tf.int32),
                "class_output": tf.TensorSpec((None,), tf.int32),
            }
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec((None,), tf.int32),
            {
                "binary_output": tf.TensorSpec((None, 1), tf.int32),
                "class_output": tf.TensorSpec((None,), tf.int32),
            }
        )
    )

    # Padding values
    input_pad = letters_id["<PAD>"]
    binary_pad = 0
    class_pad = 7  # Pad with no diactric class

    # Bucket batching configuration
    bucket_boundaries = [16, 32, 64, 96, 128, 156, 212, 256]
    bucket_batch_sizes = [256, 128, 128, 96, 96, 64, 64, 64, 32]
    # bucket_boundaries = [32, 64, 96, 128, 192, 256, 400, 500]
    # bucket_batch_sizes = [256, 128, 128, 96, 96, 64, 64, 32, 32]
    
    print(f"\nBucket configuration:")
    print(f"  Boundaries: {bucket_boundaries}")
    print(f"  Batch sizes: {bucket_batch_sizes}")

    # Shuffle training data
    train_ds = train_ds.shuffle(
        buffer_size=min(20000, len(train_inputs)),
        seed=42,
        reshuffle_each_iteration=True
    )

    # Apply bucketing to training data
    train_ds = train_ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=(
            [None],
            {
                "binary_output": [None, 1],
                "class_output": [None]
            }
        ),
        padding_values=(
            input_pad,
            {
                "binary_output": binary_pad,
                "class_output": class_pad
            }
        )
    )

    # Apply bucketing to validation data
    val_ds = val_ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[min(b * 2, 256) for b in bucket_batch_sizes],
        padded_shapes=(
            [None],
            {
                "binary_output": [None, 1],
                "class_output": [None]
            }
        ),
        padding_values=(
            input_pad,
            {
                "binary_output": binary_pad,
                "class_output": class_pad
            }
        )
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("Datasets ready!")

    # --------------------
    # Build Model
    # --------------------

    print("\nBuilding model...")
    model = BiLSTM(
        input_vocab_size=len(letters_id),  # 38
        binary_output_size=1,
        class_output_size=8
    )

    # Optimizer with gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=2e-3,
        clipnorm=1.0
    )

    # Compile model with two outputs
    model.compile(
        optimizer=BNGAdam(learning_rate=0.001, block_size=64),
        loss={
            "binary_output": keras.losses.BinaryCrossentropy(),
            "class_output": keras.losses.SparseCategoricalCrossentropy()
        },
        metrics={
            "binary_output": ["accuracy"],
            "class_output": ["accuracy"]
        }
    )

    print("Model compiled successfully!")

    # --------------------
    # Callbacks
    # --------------------

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.keras"),
            monitor="val_class_output_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
            save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"),
            append=True
        )
    ]

    # --------------------
    # Train
    # --------------------

    print(f"\nStarting training...")
    print(f"Checkpoints will be saved to: {output_dir}")
    print("=" * 60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {history_path}")
    
    # List all saved files
    print("\n" + "=" * 60)
    print("Saved files:")
    if os.path.exists(output_dir):
        for file in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")
    
    print("\nTraining complete!")