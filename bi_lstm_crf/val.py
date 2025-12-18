import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.preprocess import load_dataset
from .train import encode_dataset, PAD_LABEL
from utils.dataset_builder import letters_id
from .bi_lstm_crf import BiLSTM_CRF
from utils.config import *


if __name__ == "__main__":
    val_file = "/kaggle/input/testset/test.txt"
    val_lines = load_dataset(val_file)

    output_dir = WORKING + "checkpoints"
    cache_dir = WORKING + "cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    val_inputs, val_targets = encode_dataset(
        val_lines,
        os.path.join(cache_dir, "val_encoded.pkl"),
    )

    model_file = os.path.join(output_dir, "best_model.keras")
    if not os.path.exists(model_file):
        print(f"✗ File not found: {model_file}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print(f"Loading model from: {model_file}")
    print("=" * 60)

    model = keras.models.load_model(
        model_file,
        custom_objects={"BiLSTM_CRF": BiLSTM_CRF},
        compile=False,
    )
    model.compile(optimizer="adam")  

    print("\nComputing CRF Viterbi validation accuracy...")

    total = 0
    correct = 0

    for x, y_true in zip(val_inputs, val_targets):
        x_arr = np.array([x], dtype=np.int32)  
        y_true_arr = np.array(y_true, dtype=np.int32)

        logits = model(x_arr, training=False).numpy()

        mask = (y_true_arr != PAD_LABEL)

        pred = model.decode_batch(logits, mask[np.newaxis, :])[0]

        n = min(len(pred), len(y_true_arr))
        total += np.sum(mask[:n])
        correct += np.sum((pred[:n] == y_true_arr[:n]) & mask[:n])

    acc = correct / total if total > 0 else 0.0
    print(f"CRF Viterbi Validation Accuracy = {acc:.4f} ({acc*100:.2f}%)")

