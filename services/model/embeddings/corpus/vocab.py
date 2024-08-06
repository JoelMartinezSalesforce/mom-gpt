from typing import List
import re
from collections import Counter
from dateutil.parser import parse
import numpy as np


class VocabularyCreator:
    def __init__(self, ngram_range=(1, 2), special_characters="!@#$%^&*()_+-=[]{}|;:',.<>?"):
        """
        Initializes the VocabularyCreator with specific configurations for generating n-grams
        and handling special characters.

        Args:
            ngram_range (tuple): Specifies the range of n-grams to generate, e.g., (1, 2) for unigrams and bigrams.
            special_characters (str): A string containing all special characters to be considered for removal.
        """

        self.vocab = {}
        self.ngram_range = ngram_range
        self.special_characters = special_characters

    def create_vocab(self, preprocessed_texts: List[str], threshold: float = None):
        """
        Processes a list of preprocessed texts to create a vocabulary based on n-grams, applying
        a frequency threshold and filtering for significance.

        Args:
            preprocessed_texts (List[str]): List of preprocessed text strings.
            threshold (float, optional): Minimum frequency threshold for terms to be included in the vocabulary.
                                         If None, calculates a dynamic threshold based on mean frequencies.

        Returns:
            Dict[str, int]: A dictionary of terms with their corresponding counts, filtered by significance and threshold.
        """

        # Create n-grams
        ngrams = self.generate_ngrams(preprocessed_texts)

        # Calculate frequencies
        word_counts = Counter(ngrams)
        total_words = sum(word_counts.values())

        # Dynamic threshold if not set
        if threshold is None:
            threshold = np.mean(list(word_counts.values())) / total_words

        # Filter based on threshold and significance
        self.vocab = {word: count for word, count in word_counts.items() if
                      (count / total_words) >= threshold and self.is_significant(word)}

        # Process special characters and dates
        self.process_special_characters()
        # self.process_dates()

        return self.vocab

    def generate_ngrams(self, texts: List[str]):
        """
        Generates n-grams from a list of text strings based on the configured ngram_range.

        Args:
            texts (List[str]): List of text strings from which to generate n-grams.

        Returns:
            List[str]: List of generated n-grams.
        """

        ngrams = []
        for text in texts:
            # Ensure proper splitting by also removing unwanted special characters at this step
            words = re.sub(f"[{re.escape(self.special_characters)}]", ' ', text).split()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams.extend([' '.join(words[i:i + n]).strip() for i in range(len(words) - n + 1)])
        return ngrams

    def is_significant(self, term):
        """
        Determines if a term is significant based on its content. Terms that are purely numeric are considered
        non-significant unless they are part of a meaningful context including alphabetic characters.

        Args:
            term (str): The term to evaluate for significance.

        Returns:
            bool: True if the term is considered significant, otherwise False.
        """

        # Allow numbers if they are part of a context including alphabetic characters
        if term.isdigit():
            return False
        # Check if the term has at least some alphabetic content
        return any(char.isalpha() for char in term)

    def process_special_characters(self):
        """
        Processes the vocabulary to remove special characters from terms and adjust their counts in the vocab.

        Modifies:
            self.vocab (Dict[str, int]): The vocabulary is updated by cleaning terms of special characters.
        """

        updated_vocab = {}
        for word, count in self.vocab.items():
            cleaned_word = re.sub(f"[{re.escape(self.special_characters)}]", '', word)
            updated_vocab[cleaned_word] = updated_vocab.get(cleaned_word, 0) + count
        self.vocab = updated_vocab

    def process_dates(self):
        """
        Processes the vocabulary to format terms recognized as dates into a standard form and adjust their counts.

        Modifies:
            self.vocab (Dict[str, int]): The vocabulary is updated by formatting date terms.
        """

        updated_vocab = {}
        for word, count in self.vocab.items():
            try:
                parsed_date = parse(word, fuzzy=False)
                new_key = "DATE:" + parsed_date.strftime("%Y-%m-%d")
                updated_vocab[new_key] = updated_vocab.get(new_key, 0) + count
            except ValueError:
                updated_vocab[word] = updated_vocab.get(word, 0) + count
        self.vocab = updated_vocab
