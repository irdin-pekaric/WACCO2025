import pandas as pd
from langdetect import detect, LangDetectException
from tqdm import tqdm
import argparse
import re

# Define a function to count words in a string
def word_count(content):
    # Split the string by whitespace and count the resulting words
    return len(content.split())

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    # Remove emojis
    return text.strip()

# Function to check if the text is in English with verbose output
def is_english_test(text):
    try:
        if not isinstance(text, str):
            return False

        iseng = detect(text) == 'en'

        print("Is English : ", iseng)
        print("_______________TEXTBELOW_______________")
        print(text)
        print("_______________TEXTABOVE_______________\n\n\n\n")

        return iseng
    except LangDetectException:
        return False

# Function to check if the text is in English
def is_english(text):
    if not isinstance(text, str):
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def main(input_file, output_file, is_test):
    print(f"Processing : {input_file}")

    # Read input file
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['preprocessed_content'])

    if is_test:
        df = df[df['preprocessed_content'].apply(word_count) > 7]
        df = df.sample(n=100, random_state=42)

    # Initialize tqdm progress bar
    tqdm.pandas()

    # Filter the DataFrame to only include rows where 'content' is in English
    if is_test:
        df['is_english'] = df['preprocessed_content'].progress_apply(is_english_test)
    else:
        df['is_english'] = df['preprocessed_content'].progress_apply(is_english)

    df_english = df[df['is_english']]

    # Remove duplicate rows based on the 'content' column
    df_english = df_english.drop_duplicates(subset='preprocessed_content')

    # Save the filtered DataFrame to a new CSV file
    df_english.drop(columns=['is_english'], inplace=True)
    df_english.to_csv(output_file, index=False)

    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and preprocess text data.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to the output CSV file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    args = parser.parse_args()

    main(args.input, args.output, args.test)

