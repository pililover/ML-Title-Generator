import pandas as pd
import re
from underthesea import word_tokenize

def clean_text(text):
    if pd.isna(text):
        return text
    text = re.sub(r'[\\n\\t\\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def segment_text(text):
    return word_tokenize(text, format="text")