import pandas as pd
import re
from underthesea import word_tokenize

def clean_text(text):
    if pd.isna(text):
        return text
    text = re.sub(r'[\\n\\t\\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    return word_tokenize(text, format="text")

def preprocess_input(text):
    cleaned_text = clean_text(text)
    tokenized_text = tokenize_text(cleaned_text)
    return tokenized_text

def extract_vocab(text):
    cleaned_text = clean_text(text)
    tokenized_text = tokenize_text(cleaned_text)
    return tokenized_text.split()  # Tách từ