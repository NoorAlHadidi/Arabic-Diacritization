import pickle
import numpy as np
import os
from utils.preprocess import load_dataset
from utils.dataset_builder import (
    extract_letters_and_diacritics, letters_id, diacritics, diacritics_id
)
from utils.dataset_builder import compute_sentence_word_indices
from utils.word_embeddings import ArabicWordEmbedder
from .bi_lstm_aravec import BiLSTMAraVec, BNGAdam
from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
from utils.config import *

# create reverse mappings
idx2letter = {v: k for k, v in letters_id.items()}
idx2diacritics = {v: k for k, v in diacritics_id.items()}

def remove_diacritics(text):
    '''
    Remove all diacritics from text to get the base form.
    '''
    diac_set = {'ّ', 'ْ', 'ً', 'ُ', 'ِ', 'ٌ', 'َ', 'ٍ'}
    return ''.join([c for c in text if c not in diac_set])

def load_word_dictionary(dict_path):
    '''
    Load dictionary mapping base word -> diacritized word(s).
    '''
    if not os.path.exists(dict_path):
        print(f"Dictionary file not found: {dict_path}")
        return None
    with open(dict_path, "rb") as f:
        return pickle.load(f)

def get_best_diacritized_word(base_word, candidates):
    '''
    Choose best diacritized word among candidates.
    Kept same logic (simple first-candidate strategy).
    '''
    if not candidates:
        return None
    return candidates[0]

def apply_dictionary_correction_to_sequence(input_tensor, predicted_ids, word_dict):
    '''
    Apply dictionary correction at word level.
    '''
    if isinstance(input_tensor, dict):
        input_tensor = input_tensor["char_ids"]

    if word_dict is None:
        return predicted_ids

    # convert input ids back to string
    input_ids = [int(input_tensor[0, i].numpy()) for i in range(input_tensor.shape[1])]
    letters = [idx2letter.get(i, "") for i in input_ids]

    # split into words (space-separated)
    words = []
    current_word = []
    for ch in letters:
        if ch == " ":
            if current_word:
                words.append("".join(current_word))
                current_word = []
            words.append(" ")  # keep spaces
        else:
            current_word.append(ch)
    if current_word:
        words.append("".join(current_word))

    # decode predicted ids to diacritics string (parallel to letters)
    diacs = [idx2diacritics.get(i, "") for i in predicted_ids]

    # apply correction per word span
    corrected = predicted_ids.copy()
    pos = 0
    for w in words:
        if w == " ":
            pos += 1
            continue

        base = remove_diacritics(w)
        candidates = word_dict.get(base)
        best = get_best_diacritized_word(base, candidates)

        if best is not None:
            best_letters, best_labels = extract_letters_and_diacritics(best, diacritics, diacritics_id)
            for i in range(min(len(best_labels), len(w))):
                corrected[pos + i] = diacritics_id.get(best_labels[i], corrected[pos + i])

        pos += len(w)

    return corrected

def combine_predictions(binary_flag, class_id):
    mapping = {
        (0,0): 0, (0,1): 1, (0,2): 2, (0,3): 3,
        (0,4): 4, (0,5): 5, (0,6): 6, (0,7): 14,
        (1,0): 8, (1,1): 9, (1,2): 10, (1,3): 11,
        (1,4): 12, (1,5): 13, (1,7): 7
    }
    return mapping.get((binary_flag, class_id), 14)

def compute_char_positions_in_word(word_indices):
    positions = np.zeros((len(word_indices), 1), dtype=np.float32)

    # Collect char positions per word id
    words = {}
    for i, widx in enumerate(word_indices):
        if widx == -1:
            continue
        words.setdefault(widx, []).append(i)

    # Fill normalized positions
    for widx, idxs in words.items():
        L = len(idxs)
        if L == 1:
            positions[idxs[0]] = 0.5
        else:
            for j, pos in enumerate(idxs):
                positions[pos] = j / (L - 1)

    return positions

def line_to_input_tensor(line, word_embedder=None):
    '''
    Convert one diacritized Arabic line into model input tensor(s).
    Outputs:
        model_input: dict with keys:
            "char_ids": tf.Tensor of shape (1, sequence_length)
            "word_embeddings": tf.Tensor of shape (1, sequence_length, embedding_dim)
            "char_positions": tf.Tensor of shape (1, sequence_length, 1)
        input_ids: list of int (encoded letters)
        target_ids: list of int (encoded diacritics)
    '''
    if word_embedder is None:
        raise ValueError("word_embedder is required for BiLSTMAraVec inference/evaluation.")

    # extract letters and their diacritics
    letters, labels = extract_letters_and_diacritics(line, diacritics, diacritics_id)

    # encode letters to indices
    input_ids = [letters_id.get(l, 0) for l in letters]  # 0 for unknown letters

    # encode diacritics to indices
    target_ids = [diacritics_id.get(d, 14) for d in labels]

    word_indices = compute_sentence_word_indices(letters)
    word_emb = word_embedder.get_sequence_embeddings(letters, word_indices)  # (sequence_length, embedding_dim)
    char_pos = compute_char_positions_in_word(word_indices)  # (sequence_length, 1)

    model_input = {
        "char_ids": tf.constant([input_ids], dtype=tf.int32),
        "word_embeddings": tf.constant([word_emb.astype(np.float32)], dtype=tf.float32),
        "char_positions": tf.constant([char_pos.astype(np.float32)], dtype=tf.float32),
    }

    return model_input, input_ids, target_ids


def predict_ids(model, input_tensor, word_dict=None):
    '''
    Returns predicted 15-class diacritic sequence (list of ints)
    '''
    pred = model.predict(input_tensor, verbose=0)

    binary_pred = pred["binary_output"][0]   # (sequence_length, 1)
    class_pred  = pred["class_output"][0]    # (sequence_length, 8)

    binary_flags = (binary_pred > 0.5).astype(int).flatten()

    class_ids = np.argmax(class_pred, axis=-1)

    final_ids = [combine_predictions(int(binary), int(multi_class)) for binary, multi_class in zip(binary_flags, class_ids)]

    if word_dict is not None:
        final_ids = apply_dictionary_correction_to_sequence(input_tensor, final_ids, word_dict)

    return final_ids


def decode_diacritics(letters, predicted_ids):
    '''
    Attach predicted diacritics to letters, keeping spaces as is.
    '''
    out = []
    for ch, did in zip(letters, predicted_ids):
        if ch == " ":
            out.append(" ")
        else:
            out.append(ch + idx2diacritics.get(did, ""))
    return "".join(out)

def predict_and_decode(model, input_tensor, input_ids, word_dict=None):
    '''
    Predict ids then decode to diacritized string.
    '''
    predicted_ids = predict_ids(model, input_tensor, word_dict)
    letters = [idx2letter.get(i, "") for i in input_ids]
    predicted_text = decode_diacritics(letters, predicted_ids)
    return predicted_text, predicted_ids


def calculate_metrics(predicted_ids, target_ids):
    '''
    Compute:
      - exact match accuracy (sentence-level)
      - per diacritic accuracy (character-level)
    '''
    total_chars = len(target_ids)
    correct_exact = int(predicted_ids == target_ids)

    correct_individual_diacs = sum(int(p == t) for p, t in zip(predicted_ids, target_ids))

    per_diac_accuracy = correct_individual_diacs / total_chars if total_chars else 0.0
    exact_match_accuracy = correct_exact

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "per_diac_accuracy": per_diac_accuracy,
        "correct_exact": correct_exact,
        "total_chars": total_chars,
        "correct_individual_diacs": correct_individual_diacs,
        "total_individual_diacs": total_chars
    }


def evaluate_on_file(model, test_file, word_embedder, output_file=None, max_samples=None):
    '''
    Evaluate model on entire test file.
    '''
    print(f"Loading test data from {test_file}.")
    lines = load_dataset(test_file)

    if max_samples:
        lines = lines[:max_samples]

    all_stats = {
        "correct_exact": 0,
        "total_chars": 0,
        "correct_individual_diacs": 0,
        "total_individual_diacs": 0,
        "exact_match_accuracy": [],
        "per_diac_accuracy": []
    }

    results = []

    for idx, line in enumerate(tqdm(lines, desc="Evaluating", unit="line")):
        input_tensor, input_ids, target_ids = line_to_input_tensor(line, word_embedder)

        predicted_text, predicted_ids = predict_and_decode(model, input_tensor, input_ids)

        stats = calculate_metrics(predicted_ids, target_ids)

        all_stats["exact_match_accuracy"].append(stats["exact_match_accuracy"])
        all_stats["per_diac_accuracy"].append(stats["per_diac_accuracy"])
        all_stats["correct_exact"] += stats["correct_exact"]
        all_stats["total_chars"] += stats["total_chars"]
        all_stats["correct_individual_diacs"] += stats["correct_individual_diacs"]
        all_stats["total_individual_diacs"] += stats["total_individual_diacs"]

        results.append({
            "index": idx,
            "input": line,
            "predicted": predicted_text,
            "target_ids": target_ids,
            "predicted_ids": predicted_ids,
            "per_diac_accuracy": stats["per_diac_accuracy"],
            "exact_match": stats["exact_match_accuracy"]
        })

    overall_exact = all_stats["correct_exact"] / len(lines) if lines else 0.0
    overall_per_diac = all_stats["correct_individual_diacs"] / all_stats["total_individual_diacs"] if all_stats["total_individual_diacs"] else 0.0

    print("\n" + "="*50)
    print("Overall Evaluation Results")
    print("="*50)
    print(f"Exact match accuracy: {overall_exact:.4f}")
    print(f"Per-diacritic accuracy: {overall_per_diac:.4f}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Exact match accuracy: {overall_exact:.6f}\n")
            f.write(f"Per-diacritic accuracy: {overall_per_diac:.6f}\n")

    return {"exact_match_accuracy": overall_exact, "per_diac_accuracy": overall_per_diac}, results

def load_model_custom(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "BiLSTMAraVec": BiLSTMAraVec,
            "BNGAdam": BNGAdam,
        },
        compile=False,
    )

if __name__ == "__main__":
    print("Loading model.")
    print(f"Letter vocab size: {len(letters_id)}")

    model = None

    model_files = [
        "/kaggle/working/checkpoints_word_context/final_model.keras",
        # "checkpoints/final_model.keras",
        # "checkpoints/best_model_fixed.keras"
    ]

    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"File not found: {model_file}")
            continue

        print(f"\n{'='*50}")
        print(f"Trying: {model_file}")
        print('='*50)

        try:
            model = load_model_custom(model_file)
            print("Model loaded successfully!")
            break
        except Exception as exception:
            print(f"Loading failed: {str(exception)[:200]}")

    if model is None:
        print("\n" + "="*50)
        print("Error: Could not load model.")
        print("="*50)
        import sys
        sys.exit(1)

    word_embedder = ArabicWordEmbedder(model_path=ARAVEC_MODEL_PATH, embedding_dim=100)

    print("="*50 + "\n")

    print("Starting full evaluation.")
    overall, results = evaluate_on_file(
        model,
        VAL_DATASET_PATH, 
        word_embedder,
        output_file= WORKING + "evaluation_results.txt"
    )

    print("\nDone.")
