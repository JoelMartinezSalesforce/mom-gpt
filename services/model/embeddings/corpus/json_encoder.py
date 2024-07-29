import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from services.model.constants.embedding_const import EmbeddingConstants
from services.model.embeddings.embedding_model import EmbeddingModelWrapper


class JSONEncoder:
    def __init__(self, json_file_path, model_name: EmbeddingConstants = EmbeddingConstants.STELLA_EN_1_5B,
                 num_workers=4):
        self.json_file_path = json_file_path
        self.model_wrapper = EmbeddingModelWrapper.instance(model_name, EmbeddingConstants.FITTING_DIMENSIONS)
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
        removing stopwords, and tokenizing.
        """
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in self.stop_words])

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
