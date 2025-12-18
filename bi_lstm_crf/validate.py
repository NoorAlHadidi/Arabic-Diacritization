import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils.preprocess import read_dataset, clean_sentence
from utils.dataset_builder import letters_id, diacritics_id, extract_letters_and_diacritics, diacritics
from .bi_lstm_crf import BiLSTM_CRF
from utils.config import *

idx2letter = {v: k for k, v in letters_id.items()}
idx2diacritics = {v: k for k, v in diacritics_id.items()}

PAD_LABEL = 15


def load_model(model_path):
    """Load the trained BiLSTM-CRF model."""
    print(f"Loading model from: {model_path}")
    
    dummy_model = BiLSTM_CRF(
        input_vocab_size=len(letters_id),
        num_tags=15,
        pad_label=PAD_LABEL
    )
    
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'BiLSTM_CRF': BiLSTM_CRF,
            'crf_loss': dummy_model.crf_loss,
            'crf_viterbi_accuracy': lambda y_true, y_pred: tf.constant(0.0)  # Dummy metric for loading
        },
        compile=False  
    )
    print("Model loaded successfully!")
    return model


def prepare_input(text, letters_id):
    """
    Prepare input text for the model.
    Returns: encoded sequence as list of indices
    """
    
    letters, _ = extract_letters_and_diacritics(text, diacritics, diacritics_id)
    
    encoded = []
    for letter in letters:
        if letter in letters_id:
            encoded.append(letters_id[letter])
        else:
            encoded.append(letters_id[" "])
    
    return letters, encoded


def predict_diacritics(model, encoded_seq):
    """
    Predict diacritics for an encoded sequence.
    
    Args:
        model: trained BiLSTM-CRF model
        encoded_seq: list of letter indices
    
    Returns:
        predicted diacritic indices (list)
    """

    x = np.array(encoded_seq, dtype=np.int32)
    x = np.expand_dims(x, 0)  
    
    logits = model(x, training=False)  
    
    mask = np.ones((1, len(encoded_seq)), dtype=bool)
    
    predictions = model.decode_batch(logits.numpy(), mask)  
    
    return predictions[0].tolist()


def process_file(input_file, model, output_csv):
    """
    Process input text file and generate predictions in CSV format.
    
    Args:
        input_file: path to input .txt file
        model: trained model
        output_csv: path to output CSV file
    """
    print(f"\nProcessing file: {input_file}")
    
    lines = read_dataset(input_file)
    print(f"Loaded {len(lines)} lines")
    
    all_letters = []
    all_predictions = []
    
    for line_idx, line in enumerate(lines):
        if line_idx % 100 == 0:
            print(f"Processing line {line_idx + 1}/{len(lines)}...")
        
        cleaned_line = clean_sentence(line)
        
        if not cleaned_line.strip():
            continue
        
        letters, encoded_seq = prepare_input(cleaned_line, letters_id)
        
        if len(encoded_seq) == 0:
            continue
        
        predicted_labels = predict_diacritics(model, encoded_seq)
        
        for letter, label in zip(letters, predicted_labels):
            if letter != " ":  # Skip spaces
                all_letters.append(letter)
                all_predictions.append(label)
    
    print(f"\nTotal characters processed: {len(all_letters)}")
    
    df = pd.DataFrame({
        'ID': range(len(all_letters)),
        'label': all_predictions
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    
    print("\n📊 Prediction Statistics:")
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    
    for label_id in sorted(label_counts.index):
        count = label_counts[label_id]
        diac_str = repr(idx2diacritics.get(label_id, "?"))
        print(f"  Label {label_id:2d} ({diac_str:6s}): {count:8d} ({count/total*100:5.2f}%)")
    
    return df


def compare_with_gold(predictions_csv, gold_csv):
    """
    Compare predictions with gold standard and compute accuracy.
    
    Args:
        predictions_csv: path to predictions CSV
        gold_csv: path to gold standard CSV
    """
    print(f"\n{'='*60}")
    print("Comparing with gold standard...")
    print(f"{'='*60}")
    
    # Load both files
    pred_df = pd.read_csv(predictions_csv)
    gold_df = pd.read_csv(gold_csv)
    
    # Ensure same length
    if len(pred_df) != len(gold_df):
        print(f"⚠️  WARNING: Different lengths - Predictions: {len(pred_df)}, Gold: {len(gold_df)}")
        min_len = min(len(pred_df), len(gold_df))
        pred_df = pred_df.iloc[:min_len]
        gold_df = gold_df.iloc[:min_len]
    
    # Compute accuracy
    correct = (pred_df['label'] == gold_df['label']).sum()
    total = len(pred_df)
    accuracy = correct / total * 100
    
    print(f"\n✅ Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for label_id in sorted(gold_df['label'].unique()):
        gold_mask = gold_df['label'] == label_id
        gold_count = gold_mask.sum()
        
        if gold_count > 0:
            pred_labels = pred_df.loc[gold_mask, 'label']
            correct_class = (pred_labels == label_id).sum()
            class_acc = correct_class / gold_count * 100
            diac_str = repr(idx2diacritics.get(label_id, "?"))
            
            print(f"  Label {label_id:2d} ({diac_str:6s}): {class_acc:5.2f}% ({correct_class:6d}/{gold_count:6d})")
    
    # Confusion analysis (top errors)
    print(f"\n📊 Top 5 Confusion Pairs:")
    errors = pred_df[pred_df['label'] != gold_df['label']].copy()
    errors['gold'] = gold_df.loc[errors.index, 'label']
    error_pairs = errors.groupby(['gold', 'label']).size().sort_values(ascending=False).head(5)
    
    for (gold_label, pred_label), count in error_pairs.items():
        gold_str = repr(idx2diacritics.get(gold_label, "?"))
        pred_str = repr(idx2diacritics.get(pred_label, "?"))
        print(f"  Gold {gold_label:2d} ({gold_str:6s}) → Pred {pred_label:2d} ({pred_str:6s}): {count:6d} times")
    
    return accuracy


def diacritize_text(model, text):
    """
    Diacritize a single text string and return the result.
    
    Args:
        model: trained model
        text: input text (without diacritics)
    
    Returns:
        diacritized text string
    """
    # Clean the text
    cleaned_text = clean_sentence(text)
    
    if not cleaned_text.strip():
        return ""
    
    # Prepare input
    letters, encoded_seq = prepare_input(cleaned_text, letters_id)
    
    if len(encoded_seq) == 0:
        return ""
    
    # Predict diacritics
    predicted_labels = predict_diacritics(model, encoded_seq)
    
    # Reconstruct text with diacritics
    result = []
    for letter, label_idx in zip(letters, predicted_labels):
        result.append(letter)
        diacritic = idx2diacritics.get(label_idx, "")
        if diacritic and diacritic != "":
            result.append(diacritic)
    
    return "".join(result)


if __name__ == "__main__":
    print("=" * 60)
    print("Arabic Diacritization - Validation Script")
    print("=" * 60)
    
    model_path = "/kaggle/working/best_model.keras"
    input_file = "/kaggle/input/project-testset/dataset_no_diacritics.txt"
    output_csv = "/kaggle/working/predictions.csv"
    gold_csv = "/kaggle/input/unit-test/sample_test_set_gold.csv"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using train.py")
        exit(1)
    
    model = load_model(model_path)
    
    predictions_df = process_file(input_file, model, output_csv)
    
    print(f"\n{'='*60}")
    print("Sample Diacritization Demo:")
    print(f"{'='*60}")
    
    sample_texts = [
        "السلام عليكم",
        "الحمد لله",
        "كيف حالك"
    ]
    
    for sample in sample_texts:
        diacritized = diacritize_text(model, sample)
        print(f"\nInput:  {sample}")
        print(f"Output: {diacritized}")
    