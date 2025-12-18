import os
import numpy as np
import tensorflow as tf
from .test import load_model_custom, predict_ids, load_word_dictionary
from utils.preprocess import *
from .train import encode_dataset
from utils.dataset_builder import letters_id, diacritics_id
from utils.config import *
from .bi_lstm import BiLSTM
from keras.models import load_model
from tqdm import tqdm

def calculate_error_rates(prediction, target, original_line):
    """
    Calculate word error rate, diacritic error rate, and last letter error rate.
    Returns error counts and totals, plus lists of wrong words for debugging.
    """
    # Remove diacritics to get base characters
    ARABIC_DIACRITICS = set({'ّ', 'ْ', 'ً', 'ُ', 'ِ', 'ٌ', 'َ', 'ٍ'})
    
    def remove_diacritics(text):
        return ''.join(c for c in text if c not in ARABIC_DIACRITICS)
    
    # Split into words based on spaces in the original line
    words = original_line.split()
    
    # Track positions in prediction/target arrays
    word_errors = 0
    total_words = 0
    diacritic_errors = 0
    total_diacritics = 0
    last_letter_errors = 0
    total_last_letters = 0
    
    wrong_words = []
    correct_words = []
    
    char_idx = 0
    
    for word in words:
        if not word:
            continue
            
        total_words += 1
        word_base = remove_diacritics(word)
        word_length = len(word_base)
        
        if char_idx + word_length > len(prediction):
            break
        
        # Extract predictions and targets for this word
        word_pred = prediction[char_idx:char_idx + word_length]
        word_target = target[char_idx:char_idx + word_length]
        
        # Check if word is correct
        word_correct = np.array_equal(word_pred, word_target)
        if not word_correct:
            word_errors += 1
            # Reconstruct predicted and target words for debugging
            pred_word = reconstruct_word(word_base, word_pred)
            target_word = word
            wrong_words.append(pred_word)
            correct_words.append(target_word)
        else:
            wrong_words.append(None)
            correct_words.append(None)
        
        # Count diacritic errors in this word
        for j in range(word_length):
            total_diacritics += 1
            if word_pred[j] != word_target[j]:
                diacritic_errors += 1
        
        # Check last letter error
        if word_length > 0:
            total_last_letters += 1
            if word_pred[-1] != word_target[-1]:
                last_letter_errors += 1
        
        char_idx += word_length + 1
    
    return {
        'word_errors': word_errors,
        'total_words': total_words,
        'diacritic_errors': diacritic_errors,
        'total_diacritics': total_diacritics,
        'last_letter_errors': last_letter_errors,
        'total_last_letters': total_last_letters,
        'wrong_words': wrong_words,
        'correct_words': correct_words
    }

def reconstruct_word(base_word, diacritic_ids):
    """Reconstruct a word with diacritics from base characters and diacritic IDs."""
    idx2diacritics = {v: k for k, v in diacritics_id.items()}
    result = []
    for i, char in enumerate(base_word):
        result.append(char)
        if i < len(diacritic_ids) and diacritic_ids[i] in idx2diacritics:
            diac = idx2diacritics[diacritic_ids[i]]
            if diac:  # Don't add empty diacritic
                result.append(diac)
    return ''.join(result)

if __name__ == "__main__":
    val_file = DATASET + "test.txt"
    raw_lines = read_dataset(val_file)
    # write_file("raw_lines.txt", raw_lines)
    cleaned_lines = clean_lines(raw_lines)
    # write_file("cleaned_lines.txt", cleaned_lines)
    val_lines = split_lines_by_punctuation(cleaned_lines)
    # write_file("punctuation_split_lines.txt", punc_split_lines)
    val_lines = add_delimiters(val_lines)
    print("test data: ", val_lines)
    
    output_dir = WORKING
    cache_dir = WORKING + "cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    val_inputs, val_targets = encode_dataset(
        val_lines
    )

    model_files = [WORKING + "final_model.keras"]

    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"✗ File not found: {model_file}")
            
            
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

    
    dict_path = WORKING + "word_dictionary.pkl"
    word_dict = load_word_dictionary(dict_path)
    if word_dict:
        print("Will apply post-processing")

    # Initialize counters
    total_word_errors = 0
    total_words = 0
    total_diacritic_errors = 0
    total_diacritics = 0
    total_last_letter_errors = 0
    total_last_letters = 0
    
    # Open error output file
    error_file = os.path.join(output_dir, "validation_errors.txt")
    
    with open(error_file, 'w', encoding='utf-8') as f_err:
        with tqdm(total=len(val_inputs), desc="Validating", unit="sample") as pbar:
            for i in range(len(val_inputs)):
                input_tensor = tf.constant([val_inputs[i]], dtype=tf.int32)
                prediction = predict_ids(model, input_tensor, word_dict)

                # Calculate error rates for this line
                errors = calculate_error_rates(prediction, val_targets[i], val_lines[i])
                
                total_word_errors += errors['word_errors']
                total_words += errors['total_words']
                total_diacritic_errors += errors['diacritic_errors']
                total_diacritics += errors['total_diacritics']
                total_last_letter_errors += errors['last_letter_errors']
                total_last_letters += errors['total_last_letters']
                
                # Write errors to file if any word errors exist
                if errors['word_errors'] > 0:
                    f_err.write(f"\n{'='*80}\n")
                    f_err.write(f"Line {i+1}:\n")
                    f_err.write(f"Original:  {val_lines[i]}\n")
                    
                    # Reconstruct predicted line with markers for wrong words
                    pred_words = []
                    for j, (wrong, correct) in enumerate(zip(errors['wrong_words'], errors['correct_words'])):
                        if wrong is not None:
                            pred_words.append(f"{wrong}")  # Mark wrong words with brackets
                        else:
                            pred_words.append(val_lines[i].split()[j])
                    
                    f_err.write(f"Predicted: {' '.join(pred_words)}\n")
                    f_err.write(f"Errors: {errors['word_errors']}/{errors['total_words']} words wrong\n")

                # Update the progress bar
                wer = total_word_errors / total_words if total_words > 0 else 0
                der = total_diacritic_errors / total_diacritics if total_diacritics > 0 else 0
                ler = total_last_letter_errors / total_last_letters if total_last_letters > 0 else 0
                
                pbar.set_postfix({
                    "WER": f"{wer:.4f}",
                    "DER": f"{der:.4f}",
                    "LER": f"{ler:.4f}"
                })
                pbar.update(1)

    # Print final results
    print("\n" + "="*60)
    print("VALIDATION RESULTS:")
    print("="*60)
    print(f"Word Error Rate (WER):        {total_word_errors}/{total_words} = {total_word_errors/total_words:.4f}")
    print(f"Diacritic Error Rate (DER):   {total_diacritic_errors}/{total_diacritics} = {total_diacritic_errors/total_diacritics:.4f}")
    print(f"Last Letter Error Rate (LER): {total_last_letter_errors}/{total_last_letters} = {total_last_letter_errors/total_last_letters:.4f}")
    print(f"\nErrors written to: {error_file}")