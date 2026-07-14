import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Cleans and normalizes raw news text for ML feature extraction."""

    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline: lowercase, strip noise, tokenize, filter, lemmatize."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
        text = re.sub(r"<.*?>", " ", text)                          # HTML tags
        text = re.sub(r"\S+@\S+", " ", text)                        # emails
        text = re.sub(r"[^a-z\s]", " ", text)                       # keep only letters
        text = re.sub(r"\s+", " ", text).strip()                    # collapse whitespace

        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        tokens = [t for t in tokens if len(t) > 2]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def clean_series(self, texts):
        """Apply clean_text to a pandas Series / list of strings."""
        return [self.clean_text(t) for t in texts]
