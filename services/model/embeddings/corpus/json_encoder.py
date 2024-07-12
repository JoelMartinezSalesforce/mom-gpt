import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from services.model.constants.embedding_const import EmbeddingConstants
from services.model.embeddings.embedding_model import EmbeddingModelWrapper


def load_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


class JSONEncoder:
    def __init__(self, model_name: EmbeddingConstants = EmbeddingConstants.SALESFORCE_2_R, num_workers=4):

        self.model_wrapper = EmbeddingModelWrapper.instance(model_name)
        self.num_workers = num_workers

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in self.stop_words])

    def encode_json_data(self, json_file_path):
        data = load_json_file(json_file_path)
        preprocessed_data = []

        for item in data:
            item_string = json.dumps(item)
            processed_text = self.preprocess_text(item_string)
            if processed_text.strip():
                preprocessed_data.append(processed_text)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.model_wrapper.encode, [text]): text for text in preprocessed_data}
            embeddings = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Encoding Data"):
                result = future.result()
                embeddings.append(result)

        return embeddings
