from models.embeddings.bare_testing.embedding_model import EmbeddingModelWrapper

embedding_model = EmbeddingModelWrapper()

print("Welcome to the vector embedding generator. Type 'exit' to terminate.")
while True:
    user_input = input("Enter your text: ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
    embedding = embedding_model.process_input(user_input)
    print("Vector embedding:", embedding.tolist())
