import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from difflib import SequenceMatcher

from utils.preprocess import load_dataset, clean_sentence
from utils.dataset_builder import (
    letters_id,
    diacritics,
    diacritics_id,
    extract_letters_and_diacritics,
)
from utils.dictionary_builder import load_dictionary
from .bi_lstm_crf import BiLSTM_CRF
from utils.config import *

PAD_LABEL = 15
idx2diac = {v: k for k, v in diacritics_id.items()}


# ------------------------------------------------
# DICTIONARY CORRECTION
# ------------------------------------------------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def correct_word(pred_word, word_dict):
    # remove diacritics → base word
    base = "".join(c for c in pred_word if c not in diacritics)

    if base not in word_dict:
        return pred_word

    variants = word_dict[base]

    if pred_word in variants:
        return pred_word

    return max(variants, key=lambda v: similarity(pred_word, v))


# ------------------------------------------------
# PREPROCESSING (IDENTICAL TO TRAINING)
# ------------------------------------------------
def prepare_input(text):
    letters, _ = extract_letters_and_diacritics(
        text, diacritics, diacritics_id
    )
    char_ids = [letters_id.get(c, letters_id[" "]) for c in letters]
    return letters, char_ids


# ------------------------------------------------
# CRF PREDICTION (2 INPUTS)
# ------------------------------------------------
def predict(model, char_ids, word_emb_dim):
    x_char = np.expand_dims(np.array(char_ids, np.int32), 0)
    x_word = np.zeros(
        (1, len(char_ids), word_emb_dim),
        dtype=np.float32
    )

    logits = model((x_char, x_word), training=False)
    mask = np.ones((1, len(char_ids)), dtype=bool)

    preds = model.decode_batch(logits.numpy(), mask)
    return preds[0]


# ------------------------------------------------
# PROCESS ONE LINE (WITH DICTIONARY)
# ------------------------------------------------
def process_line(model, line, word_emb_dim, word_dict):
    line = clean_sentence(line)
    letters, char_ids = prepare_input(line)

    if not letters:
        return []

    preds = predict(model, char_ids, word_emb_dim)

    labels = []
    word = []

    for ch, lab in zip(letters, preds):
        if ch == " ":
            if word:
                # dictionary correction happens here
                corrected = correct_word("".join(word), word_dict)
                for c in corrected:
                    if c in diacritics:
                        labels.append(diacritics_id[c])
                word = []
        else:
            word.append(ch + idx2diac.get(lab, ""))

    if word:
        corrected = correct_word("".join(word), word_dict)
        for c in corrected:
            if c in diacritics:
                labels.append(diacritics_id[c])

    return labels


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    model_path = WORKING + "checkpoints/best_model.keras"
    dict_path = WORKING + "cache/word_dictionary.pkl"
    input_file = "/kaggle/input/project-testset/dataset_no_diacritics.txt"
    output_csv = "/kaggle/working/predictions.csv"

    # ---- Load model ----
    model = keras.models.load_model(
        model_path,
        custom_objects={"BiLSTM_CRF": BiLSTM_CRF},
        compile=False,
    )

    word_emb_dim = model.word_embedding_dim

    # ---- Load dictionary ----
    word_dict = load_dictionary(dict_path)

    # ---- Load dataset ----
    lines = load_dataset(input_file)

    all_labels = []

    for line in lines:
        all_labels.extend(
            process_line(model, line, word_emb_dim, word_dict)
        )

    # ---- Save CSV ----
    df = pd.DataFrame({
        "ID": range(len(all_labels)),
        "label": all_labels
    })

    df.to_csv(output_csv, index=False)

    print(f"✅ Saved predictions to {output_csv}")
    print(f"Total labels: {len(all_labels)}")
