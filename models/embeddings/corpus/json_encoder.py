import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from models.constants.embedding_const import EmbeddingConstants
from models.embeddings.embedding_model import EmbeddingModelWrapper


class JSONEncoder:
    def __init__(self, model_name: EmbeddingConstants = EmbeddingConstants.SALESFORCE_2_R, num_workers=4):
        # Initialize the embedding model wrapper
        self.model_wrapper = EmbeddingModelWrapper.instance(model_name)
        self.num_workers = num_workers  # Number of threads for parallel encoding

        # Setup stopwords for text preprocessing
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def load_json_file(self, json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in self.stop_words])

    def encode_json_data(self, json_file_path):
        data = self.load_json_file(json_file_path)
        preprocessed_data = []

        for item in data:
            item_string = json.dumps(item)
            processed_text = self.preprocess_text(item_string)
            if processed_text.strip():  # Ensure non-empty strings
                preprocessed_data.append(processed_text)

        # Use threading to encode data concurrently
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.model_wrapper.encode, [text]): text for text in preprocessed_data}
            embeddings = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Encoding Data"):
                result = future.result()
                embeddings.append(result)

        return embeddings
