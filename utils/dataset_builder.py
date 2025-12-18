from .preprocess import load_dataset

arabic_letters = {
'ر','آ','ئ','ؤ','ث','ط','ق','ن','ض','ت','د','إ','ا','ي','خ','ز',
'ج','ء','أ','غ','ل','ف','ذ','ك','و','س','ظ','ح','م','ع','ب',
'ة','ص','ى','ش','ه'
}

diacritics = {'ّ', 'ْ', 'ً', 'ُ', 'ِ', 'ٌ', 'َ', 'ٍ'}

diacritics_id = {
 'َ': 0, 'ً': 1, 'ُ': 2, 'ٌ': 3,
 'ِ': 4, 'ٍ': 5, 'ْ': 6, 'ّ': 7,
 'َّ': 8, 'ًّ': 9, 'ُّ': 10, 'ٌّ': 11,
 'ِّ': 12, 'ٍّ': 13,
 '': 14   
}

def build_vocab(arabic_letters):
    letters_id = {}

    index = 0
    for character in sorted(arabic_letters) + ["^", "$"]:
        letters_id[character] = index
        index += 1

    letters_id[" "] = index
    letters_id["<PAD>"] = index + 1

    return letters_id

letters_id = build_vocab(arabic_letters)

def extract_letters_and_diacritics(text, diacritics, diacritics_id):
    letters = []
    labels = []

    i = 0
    while i < len(text):
        character = text[i]
        
        if character in diacritics:
            i += 1
            continue

        letters.append(character)

        diactric_combo = ""
        j = i + 1
        while j < len(text) and text[j] in diacritics:
            diactric_combo += text[j]
            j += 1

        if diactric_combo in diacritics_id:
            labels.append(diactric_combo)
        else:
            labels.append("")

        i = j

    return letters, labels


def encode_letters_and_labels(letter_sequence, label_sequence, letters_id, diacritics_id):
    encoded_letters, encoded_diactrics = [], []

    for letters, labels in zip(letter_sequence, label_sequence):
        encoded_letters.append([letters_id[letter] for letter in letters])
        encoded_diactrics.append([diacritics_id[diacritic] for diacritic in labels])

    return encoded_letters, encoded_diactrics


def process_lines(lines, letters_id, diacritics, diacritics_id):
    all_letters = []
    all_labels = []

    for line in lines:
        letters, labels = extract_letters_and_diacritics(
            line, diacritics, diacritics_id
        )
        all_letters.append(letters)
        all_labels.append(labels)

    encoded_letters, encoded_diacritics = encode_letters_and_labels(
        all_letters, all_labels,
        letters_id, diacritics_id
    )
    
    return all_letters, all_labels, encoded_letters, encoded_diacritics


def compute_sentence_word_indices(letters):
    '''
    Computes word indices for each character in the input sequence (output of extract_letters_and_diacritics).
    '''
    word_indices = []
    current_word_index = -1
    in_word = False

    for character in letters:
        if character.isspace():
            in_word = False
            word_indices.append(-1) # space gets -1
        else:
            if not in_word:
                current_word_index += 1
                in_word = True
            word_indices.append(current_word_index)

    return word_indices

def process_lines_with_word_indices(lines, letters_id, diacritics, diacritics_id):
    '''
    Extended version of process_lines that also computes word indices per character.
    Returns:
        all_letters : list of list of str
        all_labels : list of list of str
        encoded_letters : list of list of int
        encoded_diacritics : list of list of int
        word_indices_per_line : list of list of int
    '''
    all_letters, all_labels, encoded_letters, encoded_diacritics = process_lines(
        lines, letters_id, diacritics, diacritics_id
    )

    word_indices_per_line = [compute_sentence_word_indices(letters) for letters in all_letters]

    for letters, word_indices_sequence in zip(all_letters, word_indices_per_line):
        assert len(letters) == len(word_indices_sequence), (
            "Mismatch between letters and word index sequence lengths: "
            f"{len(letters)} vs {len(word_indices_sequence)}"
        )

    return all_letters, all_labels, encoded_letters, encoded_diacritics, word_indices_per_line


if __name__ == "__main__":
    lines = load_dataset("dataset/train.txt")
    letters, labels, encoded_letters, encoded_diacritics = process_lines(lines, letters_id, diacritics, diacritics_id)
