from services.model.embeddings.embedding_model import EmbeddingModelWrapper
import torch

embedding_model = EmbeddingModelWrapper.instance()
embedding_record = []

print("Welcome to the vector embedding generator. Type 'exit' to terminate.")
while True:
    user_input = input("Enter your text: ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    embedding1 = embedding_model.process_input(user_input)
    embedding2 = embedding_model.process_input(user_input)

    print("Shape of embedding1:", embedding1.shape)
    print("Shape of embedding2:", embedding2.shape)

    if embedding1.dim() > 1:
        embedding1 = embedding1.squeeze()
    if embedding2.dim() > 1:
        embedding2 = embedding2.squeeze()

    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(embedding1, embedding2).item()

    embedding_record.append((embedding1, embedding2, similarity))
    print(f"Similarity between embeddings: {similarity:.4f}")

    if len(embedding_record) > 1:
        first_embedding = embedding_record[0][0]
        current_similarity = cos(embedding1, first_embedding).item()
        print("Comparison with first embedding:")
        print("Current embedding (first 5 elements):", embedding1.tolist()[:5])
        print("First embedding (first 5 elements):", first_embedding.tolist()[:5])
        print(f"Similarity with first embedding: {current_similarity:.4f}")
