"""Tokenizer for BM25 over English, Hindi (Devanagari), and Hinglish.

Python's \\w does not match Devanagari vowel signs (matras), so a plain \\w+ regex splits
Hindi words apart. The character class below adds the whole Devanagari block.
No stemming and no stopword list: both would need per-language tuning we cannot validate.
"""

import re

TOKEN = re.compile(r"[\w\u0900-\u097F]+")
# Devanagari danda and double danda are punctuation, not letters
PUNCT = str.maketrans({"।": " ", "॥": " "})


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower().translate(PUNCT))
