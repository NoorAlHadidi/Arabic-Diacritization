import os
import csv
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

def remove_diacritics(text):
    """Remove diacritics from text to get base characters."""
    ARABIC_DIACRITICS = set({'ّ', 'ْ', 'ً', 'ُ', 'ِ', 'ٌ', 'َ', 'ٍ'})
    return ''.join(c for c in text if c not in ARABIC_DIACRITICS)

if __name__ == "__main__":
    test_file = DATASET + "test.txt"
    # test_file = DATASET + "sample_test_no_diacritics.txt"
    test_lines = read_dataset(test_file)
    test_lines = clean_lines(test_lines)
    test_lines = split_lines_by_punctuation(test_lines)
    test_lines = add_delimiters(test_lines)
    
    output_dir = WORKING
    cache_dir = WORKING + "cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Encode dataset - this will only encode the input characters
    test_inputs, _ = encode_dataset(
        test_lines
    )

    model_files = [WORKING + "final_model.keras"]

    model = None
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
        
        # Strategy 2: Load weights only (h5 format)
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
        exit(1)

    dict_path = WORKING + "cache/word_dictionary.pkl"
    word_dict = load_word_dictionary(dict_path)

    # Prepare CSV output
    csv_file = os.path.join(output_dir, "predictions.csv")
    
    all_predictions = []
    letter_id = 0
    
    print("\n" + "="*60)
    print("Generating predictions...")
    print("="*60)
    
    with tqdm(total=len(test_inputs), desc="Predicting", unit="line") as pbar:
        for i in range(len(test_inputs)):
            input_tensor = tf.constant([test_inputs[i]], dtype=tf.int32)
            prediction = predict_ids(model, input_tensor, word_dict)
            
            # Get the clean line to know how many characters (excluding spaces)
            clean_line = test_lines[i]
            
            # Add predictions for each non-space character
            for j, char in enumerate(clean_line):
                if char in [" ", "^", "$"]:  # Ignore spaces, start token, and end token
                    continue
                if j < len(prediction):
                    all_predictions.append({
                        'ID': letter_id,
                        'label': int(prediction[j])
                    })
                    letter_id += 1
            
            pbar.update(1)
    
    # Write to CSV
    print(f"\nWriting predictions to: {csv_file}")
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pred in all_predictions:
            writer.writerow(pred)
    
    print(f"✓ Wrote {len(all_predictions)} predictions to {csv_file}")
    print("="*60)