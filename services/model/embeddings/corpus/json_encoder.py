import json
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from services.model.constants.embedding_const import EmbeddingConstants
from services.model.embeddings.embedding_model import EmbeddingModelWrapper


class JSONEncoder:
    def __init__(self, json_file_path, model_name: EmbeddingConstants = EmbeddingConstants.STELLA_EN_1_5B,
                 num_workers=4):
        self.json_file_path = json_file_path
        # self.model_wrapper = EmbeddingModelWrapper.instance(model_name, EmbeddingConstants.FITTING_DIMENSIONS)
        self.num_workers = num_workers

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

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

    def create_vocab(self, preprocessed_texts, threshold: float = 0.5):
        """
        Creates a robust vocabulary from a list of strings of preprocessed data using pattern mining principles.
        Special attention is given to 'POD' followed by numbers, which are always included regardless of their frequency.

        Args:
        preprocessed_texts (list[str]): A list of strings of preprocessed text data.
        threshold (float): The minimum fraction of documents a word must appear in to be included.

        Returns:
        List[str]: A list of unique and significant vocabulary terms extracted from the preprocessed texts.
        """
        word_counts = Counter()
        pods = set()  # To store unique POD identifiers

        for text in preprocessed_texts:
            # Remove special characters and split by spaces
            words = re.sub(r'[^\w\s]', '', text).split()
            # Collect POD identifiers separately
            pods.update(re.findall(r'\bpod\d+\b', text))

            # Update counts with filtered words, excluding purely numeric and typical date patterns
            filtered_words = [
                word for word in words
                if not re.fullmatch(r'\d+', word) and
                   not re.fullmatch(r'\d{4}-\d{2}-\d{2}', word)
            ]
            word_counts.update(filtered_words)

        # Determine the minimum occurrence count based on the threshold
        min_occurrence = max(1, int(len(preprocessed_texts) * threshold))
        # Filter vocabulary based on the occurrence threshold, ensuring numeric values are excluded
        filtered_vocab = [
            word for word, count in word_counts.items()
            if count >= min_occurrence and not word.isdigit()
        ]

        # Convert set of POD identifiers to a list and combine with the filtered vocabulary
        final_vocab = sorted(set(filtered_vocab) | pods)

        return final_vocab

    def preprocess_single_json(self, json_object):
        """
        Processes a single JSON object, extracting and preprocessing the text.

        Args:
        json_object (dict): A JSON object represented as a Python dictionary.

        Returns:
        str: The preprocessed text from the JSON object.
        """
        item_string = json.dumps(json_object)
        processed_text = self.preprocess_text(item_string)
        return processed_text
