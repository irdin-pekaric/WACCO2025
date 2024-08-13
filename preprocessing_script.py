import argparse
import pandas as pd
import unicodedata
import contractions
import re
import os
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import sys

# Precompile regular expressions
URL_REGEX = re.compile(r'https?:\S*')
WWW_REGEX = re.compile(r'\b(www\.[^\s]+)')
MENTION_REGEX = re.compile(r'@\S*')
HASHTAG_REGEX = re.compile(r'#\S*')
SPECIAL_CHARS_REGEX = re.compile(r'[^a-zA-Z.,!?;\s]')
WORD_WITH_NUMBERS_REGEX = re.compile(r'\b\w*\d\w*\b')
WORD_BOUNDARY_REGEX = re.compile(r'\b\w+\b')

def truncate_repeated_characters(text):
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text)

def remove_long_words(text, length_limit):
    def filter_long_words(match):
        word = match.group(0)
        return word if len(word) <= length_limit else ''

    return WORD_BOUNDARY_REGEX.sub(filter_long_words, text)

def optimized_clean_text(text, is_english, word_length_limit=27):
    try:
        text = URL_REGEX.sub('', text)
        text = WWW_REGEX.sub('', text)
        text = MENTION_REGEX.sub('', text)
        text = HASHTAG_REGEX.sub('', text)
        text = WORD_WITH_NUMBERS_REGEX.sub('', text)
        text = text.lower()
        text = truncate_repeated_characters(text)
        if is_english:
            text = expand_contractions(text)
            text = SPECIAL_CHARS_REGEX.sub(' ', text)
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = remove_long_words(text, word_length_limit)
    except Exception as e:
        print(f"Error processing text: {text}, error: {e}")
        return ""
    return text

def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        try:
            expanded_word = contractions.fix(word)
        except Exception as e:
            print(f"Error expanding contraction for word: {word}, error: {e}")
            expanded_word = word
        expanded_words.append(expanded_word)
    return ' '.join(expanded_words)

def make_content_unique(df):
    return df.drop_duplicates(subset=['preprocessed_content'])

def drop_empty_content(df):
    df['preprocessed_content'] = df['preprocessed_content'].str.strip()
    df = df.dropna(subset=['preprocessed_content'])
    df = df[df['preprocessed_content'] != '']
    return df

def get_file_names_in_directory(directory_path):
    file_names = []
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_names.append(filename)
    file_names.sort()
    return file_names

def main(input_path, output_path):
    file_name = os.path.basename(input_path)

    print(f"Processing {file_name}:", flush=True)
    preprocessed_file_name = "preproc_" + file_name

    try:
        file_df = pd.read_csv(input_path, sep=',')
    except pd.errors.EmptyDataError:
        print(f"Skipping {file_name} as it contains no data", flush=True)
        return

    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}. Skipping to the next file.", flush=True)
        return

    is_english = True  # Set as needed
    tqdm.pandas()
    file_df['preprocessed_content'] = file_df['raw_content'].progress_apply(lambda text: optimized_clean_text(text, is_english))
    file_df = make_content_unique(file_df)
    file_df = drop_empty_content(file_df)

    try:
        file_df.to_csv(output_path, index=False)
        print(f"{preprocessed_file_name} has been saved to {output_path}", flush=True)
    except Exception as e:
        print(f"Error saving {file_name}: {e}.", flush=True)
    print("Finished the preprocessing", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input directory")
    parser.add_argument('--output', type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    input_path = args.input.strip()
    output_path = args.output.strip()
    main(input_path, output_path)

