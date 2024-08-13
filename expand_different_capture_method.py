import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

# Define banned words
banned_words = ['porn', 'sex', 'child', 'drugs', 'lolita']

def extract_data_raw(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.find('title')
    title = title.string if title else None
    cleaned = soup.get_text()
    return title, cleaned

def extract_language(html_content):
    print("This function should be used only if the source html files are set up correctly")
    soup = BeautifulSoup(html_content, 'html.parser')
    html_tag = soup.find('html')
    return html_tag['lang'] if html_tag and 'lang' in html_tag.attrs else None

# Function to check if any banned words are in the text
def contains_banned_words(text, banned_words):
    text_lower = text.lower()
    return any(banned_word in text_lower for banned_word in banned_words)

# Define the function to check if a sentence is a full sentence
def is_full_sentence(text):
    text = text.strip()
    return bool(re.match(r'^[A-Z].*[.!?]$', text))

# Define the function to extract valid paragraphs
def extract_valid_paragraphs(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    
    valid_paragraphs = []
    for paragraph in paragraphs:
        text = paragraph.get_text().strip()
        
        # Skip paragraphs with banned words
        if contains_banned_words(text, banned_words):
            continue

        sentences = re.split(r'(?<=[.!?])\s+', text)
        full_sentences = [sentence for sentence in sentences if is_full_sentence(sentence)]

        if full_sentences:
            valid_paragraphs.append(' '.join(full_sentences))

    return ' '.join(valid_paragraphs)  # Join all paragraphs if needed

def main(input_path, output_path):
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Initialize a progress bar
    tqdm.pandas(desc="Processing HTML Code")

    # Apply the paragraph extraction function with progress bar
    df['paragraph_text_capture'] = df['html_code'].progress_apply(extract_valid_paragraphs)

    # Remove duplicates in the new column
    df = df.drop_duplicates(subset='paragraph_text_capture')
    df = df.dropna(subset='paragraph_text_capture')

    # Save the new DataFrame to a CSV file
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some HTML files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file')
    
    args = parser.parse_args()
    
    main(args.input, args.output)

