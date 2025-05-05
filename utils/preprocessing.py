import pandas as pd
import re
from underthesea import word_tokenize

import unicodedata

def clean_text(text):
    text = text.replace('\xa0', ' ')  # Thay thế non-breaking space
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[^\x20-\x7E\u00A0-\u1EF9\u0100-\u017F]', '', text)  # Loại bỏ ký tự không thuộc bảng Unicode mở rộng của tiếng Việt
    return text.strip()

def segment_text(text):
    return word_tokenize(text, format="text")
