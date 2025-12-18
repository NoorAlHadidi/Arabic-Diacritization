import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.preprocess import clean_sentence
from utils.dataset_builder import (
    letters_id,
    diacritics,
    diacritics_id,
    extract_letters_and_diacritics
)
from .bi_lstm_crf import BiLSTM_CRF
from utils.config import WORKING

# Reverse diacritic mapping
idx2diacritics = {v: k for k, v in diacritics_id.items()}

PAD_LABEL = 15


def diacritize_line(model, text):
    # Clean input
    text = clean_sentence(text)

    # Extract letters only
    letters, _ = extract_letters_and_diacritics(
        text, diacritics, diacritics_id
    )

    if not letters:
        return ""

    # Encode letters
    encoded = [
        letters_id.get(ch, letters_id[" "])
        for ch in letters
    ]

    x = np.array(encoded, dtype=np.int32)[None, :]  
    mask = np.ones((1, len(encoded)), dtype=bool)

    # Forward + CRF Viterbi
    logits = model(x, training=False)
    pred = model.decode_batch(logits, mask)[0]

    # Reconstruct output
    output = []
    for ch, lbl in zip(letters, pred):
        output.append(ch)
        diac = idx2diacritics.get(lbl, "")
        if diac:
            output.append(diac)

    return "".join(output)


if __name__ == "__main__":
    model_path = WORKING + "checkpoints/best_model.keras"

    print("Loading model...")
    model = keras.models.load_model(
        model_path,
        custom_objects={"BiLSTM_CRF": BiLSTM_CRF},
        compile=False
    )
    print("Model loaded.\n")

    print("Arabic Diacritization (type text and press Enter)")
    print("Press Enter on empty line to exit.")

    while True:
        text = input("\nInput: ").strip()
        if not text:
            break

        result = diacritize_line(model, text)
        print("Output:", result)
