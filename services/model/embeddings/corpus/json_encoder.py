import json
import os
import re
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from services.model.embeddings.corpus.vocab import VocabularyCreator
from utils.utils import get_project_root

class JSONEncoder:

    base_dir = get_project_root()

    def __init__(self, json_file_path=None):
        self.json_file_path = json_file_path

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

        self.creator = VocabularyCreator(ngram_range=(1, 2))

        # Load data only if the JSON file path is provided
        if json_file_path:
            self.data = self.load_json_file()
        else:
            self.data = []

    def load_json_file(self):
        """
        Load a JSON file and return the data.
        """
        try:
            with open(self.json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.json_file_path} does not exist.")
        except json.JSONDecodeError:
            raise ValueError("Error decoding the JSON file.")

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

        pods = set(re.findall(r'pod\d+', text))
        filtered_tokens.extend(pods)

        return ' '.join(filtered_tokens)

    def preprocess_for_encoding(self):
        """
        Preprocesses all JSON objects in the loaded data and encodes them using the embedding model.
        Raises ValueError if no JSON file was loaded.
        """
        if not self.data:
            raise ValueError("No data loaded. Ensure a valid JSON file path is provided.")

        preprocessed_data = [
            self.preprocess_text(json.dumps(item)).strip() for item in self.data if self.preprocess_text(json.dumps(item)).strip()
        ]
        return preprocessed_data

    def create_vocab(self, preprocessed_texts: List[str]) -> Dict[str, int]:
        """
        Creates or retrieves a vocabulary from a list of strings of preprocessed data.
        Raises ValueError if no JSON file was used to load data.
        """
        if not self.json_file_path:
            raise ValueError("JSON file path must be provided to create or retrieve vocabulary.")

        base_filename = os.path.splitext(os.path.basename(self.json_file_path))[0]
        vocab_dir = 'vocabs'
        vocab_file_path = os.path.join(vocab_dir, f'{base_filename}_vocab.json')
        os.makedirs(vocab_dir, exist_ok=True)

        if os.path.exists(vocab_file_path):
            with open(vocab_file_path, 'r') as file:
                return json.load(file)

        vocabulary = self.creator.create_vocab(preprocessed_texts)
        with open(vocab_file_path, 'w') as file:
            json.dump(vocabulary, file)

        return vocabulary

    def preprocess_single_text(self, text: List[Dict[str, any]]) -> str:
        """
        Preprocesses a single JSON object and encodes it using the embedding model.
        Raises ValueError if no JSON file was used.
        """
        if not self.json_file_path:
            raise ValueError("JSON file path must be provided to preprocess data.")

        preprocessed_data = [
            self.preprocess_text(json.dumps(item)).strip() for item in text if self.preprocess_text(json.dumps(item)).strip()
        ]
        return ' '.join(preprocessed_data)

    def get_vocab(self, collection_vocab_name: str) -> Dict[str, int]:
        """
        Retrieves a vocabulary from a given collection name.
        """
        return self.creator.get_vocabulary(collection_vocab_name)
