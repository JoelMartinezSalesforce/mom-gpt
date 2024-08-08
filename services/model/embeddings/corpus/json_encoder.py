import json
import re
from collections import Counter
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from services.model.constants.embedding_const import EmbeddingConstants
from services.model.embeddings.corpus.vocab import VocabularyCreator


class JSONEncoder:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

        self.creator = VocabularyCreator(ngram_range=(1, 2))

        self.data = self.load_json_file()

    def load_json_file(self):
        """
        Load a JSON file and return the data.
        """
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def preprocess_text(self, text):
        """
        Preprocesses text by lowering case, removing non-alphanumeric characters,
        removing stopwords, tokenizing, and including specific terms like POD identifiers.
        """
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Extract and include POD identifiers
        pods = set(re.findall(r'pod\d+', text))  # Regex to find all instances of 'POD' followed by numbers
        filtered_tokens.extend(pods)  # Add POD identifiers to the tokens list

        return ' '.join(filtered_tokens)

    def preprocess_for_encoding(self):
        """
        Preprocesses all JSON objects in the loaded data and encodes them using the embedding model.
        Uses a list comprehension to preprocess and filter out empty results.
        """
        # Preprocess data and filter out any empty or whitespace-only results in one go
        preprocessed_data = [
            processed_text
            for item in self.data
            if (processed_text := self.preprocess_text(json.dumps(item))).strip()
        ]

        return preprocessed_data

    def create_vocab(self, preprocessed_texts: List[str]) -> Dict[str, int]:
        """
        Creates a robust vocabulary from a list of strings of preprocessed data using pattern mining principles.
        Special attention is given to 'POD' followed by numbers, which are always included regardless of their frequency.

        Args:
        preprocessed_texts (list[str]): A list of strings of preprocessed text data.
        threshold (float): The minimum fraction of documents a word must appear in to be included.

        Returns:
        List[str]: A Dict of unique and significant vocabulary terms extracted from the preprocessed texts and their
        numerical frequencies.
        """

        # check if the file containing the vocabulary exists meang from the json file name we save it as the filename + vocab
        # if it does not exists we create it if it does exist we return that and save it to
        # vocabs/filename.json
        return self.creator.create_vocab(preprocessed_texts)

    def preprocess_single_text(self, text: List[Dict[str, any]]) -> str:
        """
                    Preprocesses all JSON objects in the loaded data and encodes them using the embedding model.
                    Uses a list comprehension to preprocess and filter out empty results.
                    """
        # Preprocess data and filter out any empty or whitespace-only results in one go
        preprocessed_data = [
            processed_text
            for item in text
            if (processed_text := self.preprocess_text(json.dumps(item))).strip()
        ]
        return ' '.join(preprocessed_data)
