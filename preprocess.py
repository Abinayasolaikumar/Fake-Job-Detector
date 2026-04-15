import re
from typing import List

try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore

    _HAS_NLTK = True
except Exception:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    _HAS_NLTK = False


_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def _tokenize_basic(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", text) if t]


def _try_tokenize_nltk(text: str) -> List[str] | None:
    if not _HAS_NLTK:
        return None
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return None


def _get_stopwords() -> set[str]:
    if not _HAS_NLTK:
        return set(_FALLBACK_STOPWORDS)
    try:
        return set(stopwords.words("english"))
    except Exception:
        return set(_FALLBACK_STOPWORDS)


def _get_lemmatizer():
    if not _HAS_NLTK:
        return None
    try:
        return WordNetLemmatizer()
    except Exception:
        return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    # Remove special characters/symbols (keep alphanumerics and whitespace)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    stop_words = _get_stopwords()
    tokens = _try_tokenize_nltk(text) or _tokenize_basic(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    lemmatizer = _get_lemmatizer()
    if lemmatizer is not None:
        try:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            pass

    return " ".join(tokens)


def preprocess_corpus(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]

