import re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        # MinMaxScaler (not StandardScaler) so all handcrafted features stay
        # non-negative -- required since MultinomialNB can't accept negative input.
        self.scaler = MinMaxScaler()
        self._fitted = False

    @staticmethod
    def _handcrafted_features(raw_texts):
        """Extract stylistic features from the ORIGINAL (uncleaned) text."""
        feats = []
        for text in raw_texts:
            text = text if isinstance(text, str) else ""
            length = len(text)
            words = text.split()
            n_words = max(len(words), 1)

            n_exclaim = text.count("!")
            n_question = text.count("?")
            n_upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            n_punct = sum(1 for c in text if c in "!?.,;:")
            punct_ratio = n_punct / max(length, 1)
            upper_ratio = n_upper_words / n_words
            digit_ratio = sum(c.isdigit() for c in text) / max(length, 1)

            feats.append([
                length,
                n_words,
                n_exclaim,
                n_question,
                n_upper_words,
                avg_word_len,
                punct_ratio,
                upper_ratio,
                digit_ratio,
            ])
        return np.array(feats, dtype=float)

    def fit_transform(self, cleaned_texts, raw_texts):
        tfidf_matrix = self.tfidf.fit_transform(cleaned_texts)
        handcrafted = self._handcrafted_features(raw_texts)
        handcrafted_scaled = self.scaler.fit_transform(handcrafted)
        self._fitted = True
        combined = sp.hstack([tfidf_matrix, sp.csr_matrix(handcrafted_scaled)]).tocsr()
        return combined

    def transform(self, cleaned_texts, raw_texts):
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fit before calling transform().")
        tfidf_matrix = self.tfidf.transform(cleaned_texts)
        handcrafted = self._handcrafted_features(raw_texts)
        handcrafted_scaled = self.scaler.transform(handcrafted)
        combined = sp.hstack([tfidf_matrix, sp.csr_matrix(handcrafted_scaled)]).tocsr()
        return combined

    def get_feature_names(self):
        tfidf_names = list(self.tfidf.get_feature_names_out())
        handcrafted_names = [
            "char_length", "word_count", "exclaim_count", "question_count",
            "uppercase_word_count", "avg_word_len", "punct_ratio",
            "upper_ratio", "digit_ratio",
        ]
        return tfidf_names + handcrafted_names
