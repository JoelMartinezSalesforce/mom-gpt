from typing import List
import re
from collections import Counter
from dateutil.parser import parse
import numpy as np

class VocabularyCreator:
    def __init__(self, ngram_range=(1, 2), special_characters="!@#$%^&*()_+-=[]{}|;:',.<>?"):
        self.vocab = {}
        self.ngram_range = ngram_range
        self.special_characters = special_characters

    def create_vocab(self, preprocessed_texts: List[str], threshold: float = None):
        # Create n-grams
        ngrams = self.generate_ngrams(preprocessed_texts)

        # Calculate frequencies
        word_counts = Counter(ngrams)
        total_words = sum(word_counts.values())

        # Dynamic threshold if not set
        if threshold is None:
            threshold = np.mean(list(word_counts.values())) / total_words

        # Filter based on threshold and significance
        self.vocab = {word: count for word, count in word_counts.items() if (count / total_words) >= threshold and self.is_significant(word)}

        # Process special characters and dates
        self.process_special_characters()
        self.process_dates()

        return self.vocab

    def generate_ngrams(self, texts: List[str]):
        ngrams = []
        for text in texts:
            # Ensure proper splitting by also removing unwanted special characters at this step
            words = re.sub(f"[{re.escape(self.special_characters)}]", ' ', text).split()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams.extend([' '.join(words[i:i + n]).strip() for i in range(len(words) - n + 1)])
        return ngrams

    def is_significant(self, term):
        # Allow numbers if they are part of a context including alphabetic characters
        if term.isdigit():
            return False
        # Check if the term has at least some alphabetic content
        return any(char.isalpha() for char in term)

    def process_special_characters(self):
        updated_vocab = {}
        for word, count in self.vocab.items():
            cleaned_word = re.sub(f"[{re.escape(self.special_characters)}]", '', word)
            updated_vocab[cleaned_word] = updated_vocab.get(cleaned_word, 0) + count
        self.vocab = updated_vocab

    def process_dates(self):
        updated_vocab = {}
        for word, count in self.vocab.items():
            try:
                parsed_date = parse(word, fuzzy=False)
                new_key = "DATE:" + parsed_date.strftime("%Y-%m-%d")
                updated_vocab[new_key] = updated_vocab.get(new_key, 0) + count
            except ValueError:
                updated_vocab[word] = updated_vocab.get(word, 0) + count
        self.vocab = updated_vocab
