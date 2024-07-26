import os
import numpy as np
from pymilvus import connections, Collection
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_plugin_epgt.llm_egpt import EGPT


class Response:
    def __init__(self):
        self.response_json = None

    def text(self):
        return self.response_json.get('text') if self.response_json else None


class Prompt:
    def __init__(self, prompt, options, system=None):
        self.prompt = prompt
        self.options = options
        self.system = system


# Example usage
def main(query, user_prompt):
    key = os.getenv('EPGT_API_KEY')
    org_id = os.getenv('ORG_ID')

    # Instantiate EGPT model with your API key
    egpt_model = EGPT(key=key)

    # Define prompt options
    options = EGPT.Options()
    options.org_id = org_id

    # Create a prompt object
    prompt = Prompt(
        f"Can you explain this information and each field and percentages with the actual values given the user prompt and provide at the end an interpretation of the values'{user_prompt}' : {query}",
        options)

    # Create a response object
    response = Response()

    # Execute the model to get a response
    responses = list(egpt_model.execute(prompt, stream=False, response=response, conversation=None))

    print(responses[0])


if __name__ == '__main__':
    # Connect to Milvus
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    print("Performing a single vector search...")

    COLLECTION_NAME = "health_data_cons_final"
    health_embeddings = Collection(name=COLLECTION_NAME)

    prompt = input("Your Prompt: ")

    vocabulary = [
        "period", "instance", "site", "metric", "end", "start", "updated", "percentage",
        "percentage-EUROPE", "percentage-NORTH_AMERICA", "percentage-ASIA", "power-p95",
        "power-max", "percentage-OCEANIA", "power-avg", "percentage-max", "sum", "percentage-p95",
        "max", "percentage-CHINA"
    ]

    # Set max_features to 329 to ensure exactly 329 dimensions
    vectorizer = TfidfVectorizer(max_features=329, vocabulary=vocabulary)

    # TfidfVectorizer expects a list of documents
    vector_res = vectorizer.fit_transform([prompt]).toarray()

    print(f"Embeddings: {vector_res}")
    print(f"Embeddings shape: {vector_res.shape}")

    # Directly use the embedding vectors for Milvus search
    res = health_embeddings.search(vector_res, "embeddings", search_params, limit=5, output_fields=["id", "data"])

    print(res)

    # Optionally, you can use the `main` function if needed
    main(res, prompt)
