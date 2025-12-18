import pickle
import numpy as np
import os
from utils.preprocess import load_dataset
from utils.dataset_builder import (
    extract_letters_and_diacritics, letters_id, diacritics, diacritics_id
)
from .bi_lstm import BiLSTM
from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
from utils.config import *

# Create reverse mappings
idx2letter = {v: k for k, v in letters_id.items()}
idx2diacritics = {v: k for k, v in diacritics_id.items()}

###############################################################################
# DICTIONARY UTILITIES
###############################################################################

def remove_diacritics(text):
    """Remove all diacritics from text to get the base form."""
    diac_set = {'ّ', 'ْ', 'ً', 'ُ', 'ِ', 'ٌ', 'َ', 'ٍ'}
    return ''.join([ch for ch in text if ch not in diac_set])

def calculate_edit_distance(s1, s2):
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def load_word_dictionary(dict_path):
    """Load the pre-built word dictionary."""
    if not os.path.exists(dict_path):
        print(f"Warning: Dictionary not found at {dict_path}")
        print("Dictionary correction will be disabled.")
        return None
    
    with open(dict_path, 'rb') as f:
        word_dict = pickle.load(f)
    print(f"Loaded dictionary with {len(word_dict)} base words")
    return word_dict

def only_last_char_differs(word1, word2):
    """Check if only the last character differs between two words."""
    if len(word1) != len(word2):
        return False
    if len(word1) == 0:
        return False
    
    # Check if all characters except last are the same
    return word1[:-1] == word2[:-1] and word1[-1] != word2[-1]

def apply_dictionary_correction(word, word_dict):
    """
    Apply dictionary-based correction to a word.
    
    Args:
        word: Diacritized word to potentially correct
        word_dict: Dictionary mapping {base_word: [variants]}
    
    Returns:
        Corrected word (or original if no correction needed)
    """
    if word_dict is None:
        return word
    
    # Get non-diacritized version
    base_word = remove_diacritics(word)
    
    # If word not in dictionary, keep original
    if base_word not in word_dict:
        return word
    
    variants = word_dict[base_word]
    
    # If word matches one of the dictionary variants, keep it
    if word in variants:
        return word
    
    # Find variant with smallest edit distance
    min_distance = float('inf')
    best_variant = word
    
    for variant in variants:
        # Skip correction if only last character differs
        if only_last_char_differs(word, variant):
            continue
        
        distance = calculate_edit_distance(word, variant)
        if distance < min_distance:
            min_distance = distance
            best_variant = variant
    
    return best_variant

###############################################################################
# 1. MAP 7-class + shadda flag → original 15 diacritic IDs
###############################################################################

def combine_predictions(binary_flag, class_id):
    """
    Merge:
        binary_flag ∈ {0,1}    (shadda or not)
        class_id    ∈ [0..6]   (7-class diacritic)

    Back into your original 15-class mapping:
        0  = fatha (َ)
        1  = fathatan (ً)
        2  = damma (ُ)
        3  = dammatan (ٌ)
        4  = kasra (ِ)
        5  = kasratan (ٍ)
        6  = sukun (ْ)
        7  = shadda-only (ّ)
        8  = shadda + fatha (َّ)
        9  = shadda + fathatan (ًّ)
        10 = shadda + damma (ُّ)
        11 = shadda + dammatan (ٌّ)
        12 = shadda + kasra (ِّ)
        13 = shadda + kasratan (ٍّ)
        14 = EMPTY
    """

    # If no shadda
    if binary_flag == 0:
        base_map = {
            0: 0,  # fatha
            1: 1,  # fathatan
            2: 2,  # damma
            3: 3,  # dammatan
            4: 4,  # kasra
            5: 5,  # kasratan
            6: 6,  # sukun
            7: 14  # empty
        }
        return base_map.get(class_id, 14)  # Return 14 (EMPTY) if class_id not in map

    # If shadda exists
    shadda_map = {
        0: 8,   # shadda + fatha
        1: 9,   # shadda + fathatan
        2: 10,  # shadda + damma
        3: 11,  # shadda + dammatan
        4: 12,  # shadda + kasra
        5: 13,  # shadda + kasratan
        6: 6,   # shadda-only
        7: 14    # empty -> shadda-only
    }
    return shadda_map.get(class_id, 7)  # Return 7 (shadda-only) if class_id not in map


###############################################################################
# 2. Convert line → model input
###############################################################################

def line_to_input_tensor(line):
    """
    Convert a diacritized Arabic line into:
    - input tensor (letter indices)
    - input_ids (for decoding back)
    - target_ids (ground truth diacritic indices)
    """
    # Extract letters and their diacritics
    letters, labels = extract_letters_and_diacritics(line, diacritics, diacritics_id)
    
    # Encode letters to indices with error handling
    input_ids = [letters_id.get(l, 0) for l in letters]  # Use 0 as default for unknown letters
    
    # Encode diacritics to indices
    target_ids = [diacritics_id.get(d, 14) for d in labels]  # Use 14 (EMPTY) as default
    
    # Create tensor
    input_tensor = tf.constant([input_ids], dtype=tf.int32)
    
    return input_tensor, input_ids, target_ids


###############################################################################
# 3. Merge model output → final predicted diacritic IDs
###############################################################################

def predict_ids(model, input_tensor, word_dict=None):
    """
    Returns predicted 15-class diacritic sequence (list of ints)
    """
    pred = model.predict(input_tensor, verbose=0)

    pred_binary = pred["binary_output"][0]   # (seq_len, 1)
    pred_class  = pred["class_output"][0]    # (seq_len, 8)

    # Get binary flags (0 or 1 for shadda)
    binary_flags = (pred_binary > 0.5).astype(int).flatten()
    
    # Get class predictions (0-7)
    class_ids = np.argmax(pred_class, axis=-1)

    # Combine to get final 15-class predictions
    final_ids = [combine_predictions(int(b), int(c))
                 for b, c in zip(binary_flags, class_ids)]
    
    # Apply dictionary correction if provided
    if word_dict is not None:        
        final_ids = apply_dictionary_correction_to_sequence(
            input_tensor, final_ids, word_dict
        )

    return final_ids

def apply_dictionary_correction_to_sequence(input_tensor, predicted_ids, word_dict):
    """
    Apply dictionary correction to entire sequence by splitting into words.
    Preserves exact character positions including delimiters and spaces.
    """
    # Decode to text first
    input_ids = [int(input_tensor[0, i].numpy()) for i in range(input_tensor.shape[1])]
    decoded_text = decode_predicted_line(input_ids, predicted_ids)
    
    # Track current position in the sequence
    corrected_ids = list(predicted_ids)  # Copy the original IDs
    
    # Find word boundaries while preserving structure
    current_pos = 0
    
    # Skip start delimiter if present
    if current_pos < len(input_ids) and idx2letter.get(input_ids[current_pos]) == '^':
        current_pos += 1
    
    # Process until we hit the end delimiter or end of sequence
    while current_pos < len(input_ids):
        current_char = idx2letter.get(input_ids[current_pos], '')
        
        # Stop at end delimiter
        if current_char == '$':
            break
        
        # Skip spaces
        if current_char == ' ':
            current_pos += 1
            continue
        
        # Extract word starting at current_pos
        word_start = current_pos
        word_chars = []
        word_diacs = []
        
        # Collect word characters and their diacritics
        while current_pos < len(input_ids):
            char = idx2letter.get(input_ids[current_pos], '')
            if char in [' ', '$', '^'] or char == '':
                break
            word_chars.append(char)
            word_diacs.append(idx2diacritics.get(corrected_ids[current_pos], ''))
            current_pos += 1
        
        # Reconstruct word with current diacritics
        original_word = ''.join([c + d for c, d in zip(word_chars, word_diacs)])
        
        # Apply dictionary correction to this word
        corrected_word = apply_dictionary_correction(original_word, word_dict)
        
        # If word was corrected, update the diacritic IDs
        if corrected_word != original_word:
            # Extract diacritics from corrected word
            _, corrected_labels = extract_letters_and_diacritics(
                corrected_word, diacritics, diacritics_id
            )
            
            # Update the corrected_ids at the same positions
            for i, label in enumerate(corrected_labels):
                if word_start + i < len(corrected_ids):
                    corrected_ids[word_start + i] = diacritics_id.get(label, 14)
    
    return corrected_ids

###############################################################################
# 4. Convert predicted diacritic IDs → readable Arabic diacritized text
###############################################################################

def decode_predicted_line(input_ids, predicted_ids):
    """
    Convert predicted diacritic IDs back to text with diacritics.
    
    Args:
        input_ids: List of letter indices
        predicted_ids: List of predicted diacritic indices (15-class)
    
    Returns:
        Diacritized Arabic text string
    """
    output = ""

    for letter_id, diac_id in zip(input_ids, predicted_ids):
        # Get the letter
        letter = idx2letter.get(letter_id, "")
        
        # Get the diacritic string
        diacritic = idx2diacritics.get(diac_id, "")
        
        # Combine
        output += letter + diacritic

    return output


###############################################################################
# 5. Metric Calculation
###############################################################################

def calculate_metrics(predictions, targets):
    """
    Calculate accuracy metrics.
    
    Args:
        predictions: List of predicted 15-class IDs
        targets: List of true 15-class IDs
    
    Returns:
        Dictionary with accuracy metrics
    """

    if len(predictions) != len(targets):
        n = min(len(predictions), len(targets))
        predictions = predictions[:n]
        targets = targets[:n]

    if len(targets) == 0:
        return {
            "exact_match_accuracy": 0.0,
            "per_diac_accuracy": 0.0,
            "correct_exact": 0,
            "total_chars": 0,
            "correct_individual_diacs": 0,
            "total_individual_diacs": 0
        }

    # Exact ID match (strict)
    correct_exact = sum(1 for p, t in zip(predictions, targets) if p == t)
    exact_acc = correct_exact / len(targets)

    # Per-diacritic accuracy (more lenient)
    # Compare the actual diacritic strings
    correct_parts = 0
    total_parts = 0

    for p, t in zip(predictions, targets):
        pred_diac = idx2diacritics.get(p, "")
        true_diac = idx2diacritics.get(t, "")
        
        # Convert to sets of individual characters
        pred_set = set(pred_diac) if pred_diac else set()
        true_set = set(true_diac) if true_diac else set()
        
        if true_set:
            total_parts += len(true_set)
            correct_parts += len(pred_set & true_set)

    per_diac_acc = correct_parts / total_parts if total_parts > 0 else 0.0

    return {
        "exact_match_accuracy": exact_acc,
        "per_diac_accuracy": per_diac_acc,
        "correct_exact": correct_exact,
        "total_chars": len(targets),
        "correct_individual_diacs": correct_parts,
        "total_individual_diacs": total_parts
    }


###############################################################################
# 6. Predict + Decode One Line
###############################################################################

def predict_and_decode(model, input_tensor, input_ids):
    """
    Predict diacritics and decode to text.
    """
    predicted_ids = predict_ids(model, input_tensor)
    predicted_text = decode_predicted_line(input_ids, predicted_ids)
    return predicted_text, predicted_ids


###############################################################################
# 7. Full File Evaluation
###############################################################################

def evaluate_on_file(model, test_file, output_file=None, max_samples=None):
    """
    Evaluate model on entire test file.
    """
    print(f"Loading test data from {test_file}...")
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

    print(f"Evaluating {len(lines)} lines...")
    for idx, line in enumerate(tqdm(lines)):
        try:
            # Skip empty lines
            if not line or not line.strip():
                continue
                
            # Get input and targets
            input_tensor, input_ids, target_ids = line_to_input_tensor(line)
            
            # Skip if no valid data
            if len(input_ids) == 0 or len(target_ids) == 0:
                continue

            # Predict
            predicted_text, predicted_ids = predict_and_decode(model, input_tensor, input_ids)

            # Calculate metrics
            stats = calculate_metrics(predicted_ids, target_ids)

            # Accumulate
            all_stats["exact_match_accuracy"].append(stats["exact_match_accuracy"])
            all_stats["per_diac_accuracy"].append(stats["per_diac_accuracy"])
            all_stats["correct_exact"] += stats["correct_exact"]
            all_stats["total_chars"] += stats["total_chars"]
            all_stats["correct_individual_diacs"] += stats["correct_individual_diacs"]
            all_stats["total_individual_diacs"] += stats["total_individual_diacs"]

            results.append({
                "index": idx,
                "original": line,
                "predicted": predicted_text,
                "exact_match_accuracy": stats["exact_match_accuracy"],
                "per_diac_accuracy": stats["per_diac_accuracy"]
            })
            
        except Exception as e:
            print(f"\nError on line {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Check if we have any results
    if len(results) == 0:
        print("\nNo valid samples processed!")
        return None, []

    # Overall metrics
    overall = {
        "avg_exact_match_accuracy": np.mean(all_stats["exact_match_accuracy"]) if all_stats["exact_match_accuracy"] else 0.0,
        "avg_per_diac_accuracy": np.mean(all_stats["per_diac_accuracy"]) if all_stats["per_diac_accuracy"] else 0.0,
        "overall_exact_match_accuracy":
            all_stats["correct_exact"] / all_stats["total_chars"] if all_stats["total_chars"] > 0 else 0.0,
        "overall_per_diac_accuracy":
            all_stats["correct_individual_diacs"] / all_stats["total_individual_diacs"] if all_stats["total_individual_diacs"] > 0 else 0.0,
        "total_samples": len(lines),
        "processed_samples": len(results)
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {overall['total_samples']}")
    print(f"Processed samples: {overall['processed_samples']}")
    print(f"\nExact Match Accuracy:")
    print(f"  Average: {overall['avg_exact_match_accuracy']:.4f} ({overall['avg_exact_match_accuracy']*100:.2f}%)")
    print(f"  Overall: {overall['overall_exact_match_accuracy']:.4f} ({overall['overall_exact_match_accuracy']*100:.2f}%)")
    print(f"\nPer-Diacritic Accuracy:")
    print(f"  Average: {overall['avg_per_diac_accuracy']:.4f} ({overall['avg_per_diac_accuracy']*100:.2f}%)")
    print(f"  Overall: {overall['overall_per_diac_accuracy']:.4f} ({overall['overall_per_diac_accuracy']*100:.2f}%)")
    print("="*60 + "\n")

    # Save detailed results if requested
    if output_file:
        print(f"Saving detailed results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Average Exact Match Accuracy: {overall['avg_exact_match_accuracy']:.4f}\n")
            f.write(f"Overall Exact Match Accuracy: {overall['overall_exact_match_accuracy']:.4f}\n")
            f.write(f"Average Per-Diacritic Accuracy: {overall['avg_per_diac_accuracy']:.4f}\n")
            f.write(f"Overall Per-Diacritic Accuracy: {overall['overall_per_diac_accuracy']:.4f}\n")
            f.write(f"Processed: {overall['processed_samples']}/{overall['total_samples']} samples\n\n")
            f.write("="*80 + "\n\n")
            
            for result in results:
                f.write(f"Sample {result['index']}:\n")
                f.write(f"Original:  {result['original']}\n")
                f.write(f"Predicted: {result['predicted']}\n")
                f.write(f"Exact Match: {result['exact_match_accuracy']:.4f}, Per-Diac: {result['per_diac_accuracy']:.4f}\n")
                f.write("-"*80 + "\n")
        
        print(f"Results saved!")

    return overall, results


###############################################################################
# 8. CUSTOM MODEL LOADER
###############################################################################

def load_model_custom(model_path):
    """
    Custom model loader that rebuilds the model from saved weights.
    """
    import zipfile
    import tempfile
    import os
    
    print(f"Attempting custom loading from: {model_path}")
    
    # Create new model instance
    model = BiLSTM(
        input_vocab_size=len(letters_id),
        embedding_dim=128,
        hidden_size=250,
        binary_output_size=1,
        class_output_size=7
    )
    
    # Build model with dummy data
    dummy_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
    _ = model(dummy_input)
    
    print("Model structure created, extracting weights from .keras file...")
    
    # Extract and load weights from .keras file (which is a zip)
    try:
        with zipfile.ZipFile(model_path, 'r') as z:
            # List contents
            files = z.namelist()
            print(f"Found {len(files)} files in archive")
            
            # Extract to temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extractall(tmpdir)
                
                # Try to load weights from the extracted files
                weights_path = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(weights_path):
                    print("Loading from model.weights.h5...")
                    model.load_weights(weights_path)
                    return model
                else:
                    # Try alternative weight loading
                    print("Trying alternative weight extraction...")
                    # Look for .weights directory
                    weights_dir = os.path.join(tmpdir, 'variables')
                    if os.path.exists(weights_dir):
                        print(f"Loading from variables directory...")
                        # This is a Keras 3 format, need different approach
                        raise Exception("Keras 3 format detected, use TensorFlow checkpoint loader")
                    
    except Exception as e:
        print(f"Custom loading failed: {e}")
        raise
    
    return model

def single_sentence_test(test_sentence):
    print("\n" + "="*60)
    print("SINGLE SENTENCE TEST")
    print("="*60)
    
    try:
        input_tensor, input_ids, _ = line_to_input_tensor(test_sentence)
        predicted_text, _ = predict_and_decode(model, input_tensor, input_ids)
        print(f"Input:     {test_sentence}")
        print(f"Predicted: {predicted_text}")
    except Exception as e:
        print(f"Error in single sentence test: {e}")
        import traceback
        traceback.print_exc()

###############################################################################
# 9. MAIN
###############################################################################

if __name__ == "__main__":
    print("Loading model...")
    print(f"Letter vocab size: {len(letters_id)}")
    
    model = None
    model_files = [
        "/kaggle/working/checkpoints/best_model.keras",
        # "checkpoints/final_model.keras", 
        # "checkpoints/best_model_fixed.keras"
    ]
    
    # Try each model file
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"✗ File not found: {model_file}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Trying: {model_file}")
        print('='*60)
        
        # Strategy 1: Standard load with compile=False
        try:
            print("Strategy 1: Standard load...")
            model = load_model(
                model_file,
                custom_objects={"BiLSTM": BiLSTM},
                compile=False
            )
            print("✓ Model loaded successfully!")
            break
        except Exception as e:
            print(f"✗ Strategy 1 failed: {str(e)[:100]}")
        
        # Strategy 2: Load weights only (TensorFlow SavedModel format)
        try:
            print("\nStrategy 2: Create model and load weights (h5 format)...")
            model = BiLSTM(
                input_vocab_size=len(letters_id),
                embedding_dim=128,
                hidden_size=250,
                binary_output_size=1,
                class_output_size=7
            )
            
            # Build model
            dummy_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
            _ = model(dummy_input)
            
            # Try loading as h5 weights
            h5_file = model_file.replace('.keras', '.h5')
            if os.path.exists(h5_file):
                model.load_weights(h5_file)
                print(f"✓ Loaded weights from: {h5_file}")
                break
            else:
                print(f"✗ H5 file not found: {h5_file}")
        except Exception as e:
            print(f"✗ Strategy 2 failed: {str(e)[:100]}")
        
        # Strategy 3: Custom loader
        try:
            print("\nStrategy 3: Custom weight extraction...")
            model = load_model_custom(model_file)
            print("✓ Model loaded with custom loader!")
            break
        except Exception as e:
            print(f"✗ Strategy 3 failed: {str(e)[:100]}")
    
    if model is None:
        print("\n" + "="*60)
        print("ERROR: Could not load model with any strategy!")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Check if any .keras or .h5 files exist in checkpoints/")
        print("2. The model architecture might have changed since training")
        print("3. Try re-training with: python train.py")
        print("\nListing checkpoints directory:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                fpath = os.path.join("checkpoints", f)
                size = os.path.getsize(fpath) / (1024*1024)
                print(f"  - {f} ({size:.2f} MB)")
        else:
            print("  ✗ checkpoints/ directory not found!")
        import sys
        sys.exit(1)

    single_sentence_test("ولو ترى إذ وقفوا على النار فقالوا ياليتنا نرد ولا نكذب بئايات ربنا ونكون من المؤمنين")
    
    print("="*60 + "\n")

    # Evaluate full file
    print("Starting full evaluation...")
    overall, results = evaluate_on_file(
        model,
        "/kaggle/input/arabic/val.txt",  # Changed from "../dataset/val.txt"
        output_file="/kaggle/working/evaluation_results.txt",
        max_samples=None  # Set to a number for quick testing, None for full eval
    )
    
    if overall:
        print("\nEvaluation complete!")
    else:
        print("\nEvaluation failed - no valid samples processed.")