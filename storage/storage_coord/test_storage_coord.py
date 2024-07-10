from models.constants.embedding_const import EmbeddingConstants
from models.data.gen.data_generator import MockDataGenerator
from models.embeddings.embedding_model import EmbeddingModelWrapper

if __name__ == '__main__':
    embedding = EmbeddingModelWrapper(EmbeddingConstants.SALESFORCE_2_R)
    data = MockDataGenerator()

    data.create_new_dump(2)

    element = data.recall_data_dump()[0]

    embedding = embedding.process_input(str(element))