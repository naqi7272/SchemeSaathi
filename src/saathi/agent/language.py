"""Deterministic language detection: en, hi (Devanagari), or hinglish (Roman-script Hindi).

Rule: mostly Devanagari letters -> hi. Otherwise count common Hindi function words written
in Roman script; two or more distinct hits (or one in a very short message) -> hinglish.
A word list is crude but transparent and testable; misses are measured in the agent eval.
"""

import re

from saathi.schemas import Language

HINGLISH_WORDS = {
    "hai",
    "hain",
    "ho",
    "hoga",
    "hogi",
    "kya",
    "kaise",
    "kitna",
    "kitni",
    "kitne",
    "kab",
    "kahan",
    "kaun",
    "mein",
    "main",
    "mera",
    "meri",
    "mere",
    "mujhe",
    "hum",
    "humara",
    "aap",
    "aapka",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "nahi",
    "nahin",
    "bhi",
    "aur",
    "ya",
    "milega",
    "milegi",
    "milta",
    "milti",
    "chahiye",
    "sakta",
    "sakti",
    "sakte",
    "kar",
    "karna",
    "karein",
    "saal",
    "umar",
    "paisa",
    "paise",
    "yojana",
    "liye",
    "wala",
    "wali",
    "batao",
    "bataiye",
    "kripya",
    "abhi",
    "toh",
    "lekin",
    "agar",
}
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[a-z]+")


def detect_language(text: str) -> Language:
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    if dev > lat:
        return Language.HI
    words = WORD.findall(text.lower())
    hits = {w for w in words if w in HINGLISH_WORDS}
    if len(hits) >= 2 or (hits and len(words) <= 4):
        return Language.HINGLISH
    return Language.EN
