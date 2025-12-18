import pickle
import os
from collections import defaultdict
from .preprocess import load_dataset
from .config import *

def remove_diacritics(text):
    '''
    Remove Arabic diacritics from the text to get base form.
    '''
    diac_set = {'ّ', 'ْ', 'ً', 'ُ', 'ِ', 'ٌ', 'َ', 'ٍ'}
    return ''.join([ch for ch in text if ch not in diac_set])

def build_word_dictionary(lines):
    '''
    Build a dictionary mapping non-diacritized words to all their diacritized variants.
    
    Returns:
        dict: {non_diacritized_word: set(diacritized_variants)}
    '''
    word_dict = defaultdict(set)
    
    print(f"Building dictionary from {len(lines)} lines...")
    
    for line in lines:
        # remove start/end delimiters if present
        line = line.strip('^$')
        
        # split into words
        words = line.split()
        
        for word in words:
            if not word:
                continue
            
            # get non-diacritized version
            base_word = remove_diacritics(word)
            
            # add diacritized variant
            word_dict[base_word].add(word)
    
    # convert sets to lists for easier serialization
    word_dict = {k: list(v) for k, v in word_dict.items()}
    
    print(f"Dictionary built: {len(word_dict)} unique base words")
    print(f"Total variants: {sum(len(v) for v in word_dict.values())}")
    
    return word_dict

def save_dictionary(word_dict, filepath):
    '''
    Save dictionary to pickle file.
    '''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(word_dict, f)
    print(f"Dictionary saved to: {filepath}")

def load_dictionary(filepath):
    '''
    Load dictionary from pickle file.
    '''
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # build dictionary from training data
    train_file = DATASET + "train.txt"
    test_file = DATASET + "val.txt"
    
    print("Loading training data...")
    train_lines = load_dataset(train_file)
    test_lines = load_dataset(test_file)
    all_lines = train_lines + test_lines
    
    print(f"Total lines: {len(all_lines)}")
    
    # build dictionary
    word_dict = build_word_dictionary(all_lines)
    
    # save dictionary
    dict_path = WORKING + "cache/word_dictionary.pkl"
    save_dictionary(word_dict, dict_path)
    
    # show some statistics
    print("\nSample entries:")
    for i, (base, variants) in enumerate(list(word_dict.items())[:5]):
        print(f"  '{base}' -> {len(variants)} variants")
        for v in variants[:3]:
            print(f"    - {v}")
        if len(variants) > 3:
            print(f"    ... and {len(variants)-3} more")
    
    print("\nDictionary building complete!")