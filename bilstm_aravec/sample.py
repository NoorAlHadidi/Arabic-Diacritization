import os
import csv
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .test import load_model_custom, load_word_dictionary, predict_ids, compute_char_positions_in_word
from utils.preprocess import read_dataset, clean_lines, split_lines_by_punctuation, add_delimiters
from utils.dataset_builder import process_lines_with_word_indices, letters_id, diacritics, diacritics_id
from utils.word_embeddings import ArabicWordEmbedder
from utils.config import *

def build_inputs_for_lines(lines, word_embedder):
    '''
    Build model inputs for a list of lines, using the provided word embedder.
    Inputs:
        lines: list of strings (input lines)
        word_embedder: ArabicWordEmbedder instance
    Outputs:
        model_inputs: list of dicts, each with keys:
            - "char_ids": np.array of shape (sequence_length,)
            - "word_embeddings": np.array of shape (sequence_length, embedding_dim)
            - "char_positions": np.array of shape (sequence_length,)    
    '''
    all_letters, _, encoded_letters, _, word_indices = process_lines_with_word_indices(
        lines, letters_id, diacritics, diacritics_id
    )

    # AraVec embeddings broadcast per character
    word_embs = word_embedder.get_batch_embeddings(all_letters, word_indices)

    # character position feature
    char_pos = [compute_char_positions_in_word(word_index) for word_index in word_indices]

    model_inputs = []
    for char_ids, word_embedding, char_position in zip(encoded_letters, word_embs, char_pos):
        model_inputs.append({
            "char_ids": np.asarray(char_ids, dtype=np.int32),
            "word_embeddings": np.asarray(word_embedding, dtype=np.float32),
            "char_positions": np.asarray(char_position, dtype=np.float32),
        })

    return model_inputs


if __name__ == "__main__":          
    test_lines = read_dataset(TEST_DATASET_PATH)
    test_lines = clean_lines(test_lines)
    test_lines = split_lines_by_punctuation(test_lines)
    test_lines = add_delimiters(test_lines)

    cache_dir = "/kaggle/working/cache"
    os.makedirs(WORKING, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    model_path = BILSTM_ARAVEC_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model_custom(model_path)
    print("Model loaded successfully:", model_path)

    word_dict = load_word_dictionary(DICTIONARY_INPUT_PATH)

    word_embedder = ArabicWordEmbedder(model_path=ARAVEC_MODEL_PATH, embedding_dim=100)

    model_inputs = build_inputs_for_lines(test_lines, word_embedder)

    all_predictions = []
    letter_id = 0

    print("\n" + "=" * 50)
    print("Generating Predictions.")
    print("=" * 50)

    with tqdm(total=len(model_inputs), desc="Predicting", unit="line") as pbar:
        for i in range(len(model_inputs)):
            inp = model_inputs[i]
            input_tensor = {
                "char_ids": tf.constant([inp["char_ids"]], dtype=tf.int32),
                "word_embeddings": tf.constant([inp["word_embeddings"]], dtype=tf.float32),
                "char_positions": tf.constant([inp["char_positions"]], dtype=tf.float32),
            }

            prediction = predict_ids(model, input_tensor, word_dict)

            clean_line = test_lines[i]

            for j, char in enumerate(clean_line):
                if char in [" ", "^", "$"]:
                    continue
                all_predictions.append([letter_id, int(prediction[j])])
                letter_id += 1

            pbar.update(1)

    with open(PREDICTIONS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        writer.writerows(all_predictions)

    print(f"\nDone. Predictions saved to: {PREDICTIONS_CSV_PATH}")