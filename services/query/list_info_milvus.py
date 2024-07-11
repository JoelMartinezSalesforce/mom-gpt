from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    model,
)
import random
import json

if __name__ == '__main__':

    sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2',  # Specify the model name
        device='cpu'
    )

    # Connect to Milvus
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    # Print schema and list of collections
    print("Collections in the system:", utility.list_collections())
