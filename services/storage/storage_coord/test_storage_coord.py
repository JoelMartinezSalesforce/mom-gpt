from services.model.constants.embedding_const import EmbeddingConstants
from services.storage.gen.data_generator import MockDataGenerator
from services.model.embeddings.embedding_model import EmbeddingModelWrapper

if __name__ == '__main__':
    embedding = EmbeddingModelWrapper(EmbeddingConstants.SALESFORCE_2_R)
    data = MockDataGenerator()

    data.create_new_dump(2)

    element = data.recall_data_dump()[0]
