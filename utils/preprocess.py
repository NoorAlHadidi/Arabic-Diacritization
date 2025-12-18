import re

def read_dataset(file_path):
    '''
    Reads a text dataset where each line is a sentence.
    '''
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def clean_sentence(text):
    '''
    Cleans Arabic sentence by removing unwanted characters.
    '''
    # Remove HTML tags
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub('', text)

    # Remove English letters and numbers or unwanted symbols
    non_arabic_pattern = re.compile(r'[A-Za-z0-9"`~–\u200F\ufeff\'\(\)\[\]\{\}\\/…_%#]+')
    text = non_arabic_pattern.sub('', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_lines(lines):
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(clean_sentence(line))
    return cleaned_lines


def split_lines_by_punctuation(lines):
    # Arabic + English punctuation to split on
    split_pattern = r"[،؛:؟!;,«»*\.\-]"

    processed_lines = []

    for line in lines:
        raw_chunks = re.split(split_pattern, line)

        for c in raw_chunks:
            if c is None:
                continue
            c = c.strip()
            if c:
                processed_lines.append(c)

    return processed_lines


def add_delimiters(lines):
    return [f"^{line}$" for line in lines]


def split_long_lines(dataset_lines, max_seq_len, stride=None):
    '''
    Split Arabic sentences into smaller sentences using a sliding window approach by words, where max_seq_len is a character limit
    '''
    new_lines = []

    for line in dataset_lines:
        words = line.split()
        num_words = len(words)
        if num_words == 0:
            continue

        start = 0
        while start < num_words:
            current_words = []
            current_length = 0
            i = start

            while i < num_words:
                word = words[i]
                added_length = len(word) + (1 if current_words else 0)  # space if not first
                if current_length + added_length > max_seq_len:
                    break
                current_words.append(word)
                current_length += added_length
                i += 1

            if not current_words:
                # if a single word is longer than max_seq_len, still keep it
                new_lines.append(words[start])
                i = start + 1
            else:
                new_lines.append(" ".join(current_words))

            # if all words processed, break
            if i >= num_words:
                break

            # stride in words
            stride_words = (max(1, len(current_words) // 2) if stride is None else stride)
            start += stride_words

    return new_lines


def write_file(file_name, lines):
    with open(file_name, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


def load_dataset(file_path):
    raw_lines = read_dataset(file_path)
    # write_file("raw_lines.txt", raw_lines)

    cleaned_lines = clean_lines(raw_lines)
    # write_file("cleaned_lines.txt", cleaned_lines)

    punc_split_lines = split_lines_by_punctuation(cleaned_lines)
    # write_file("punctuation_split_lines.txt", punc_split_lines)

    delimited_lines = add_delimiters(punc_split_lines)
    # write_file("delimited_lines.txt", delimited_lines)

    final_lines = split_long_lines(delimited_lines, 80, 5)
    # write_file("length_split_lines.txt", final_lines)

    return final_lines


if __name__ == "__main__":
    dataset_path = "dataset/train.txt" 
    final_lines = load_dataset(dataset_path)