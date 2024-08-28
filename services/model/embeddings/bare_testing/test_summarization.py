import spacy
import pytextrank
from icecream import ic

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.summarization.summarizer import NLPSummarization

if __name__ == "__main__":
    # Initialize the JSONEncoder class (assuming you have this setup)
    encoder = JSONEncoder(
        json_file_path="/Users/joel.martinez/Documents/mom-gpt-master/services/model/embeddings/bare_testing/dump/loodbalancer_pool.json"
    )

    text_list = encoder.preprocess_for_encoding()  # Assuming this returns a list of strings

    # Initialize the NLPSummarization class
    summarizer = NLPSummarization(10)

    # Find the longest element in the list
    longest_text = summarizer.find_longest_text(text_list)

    # Define your custom vocabulary of important keywords
    custom_keywords = list(encoder.create_vocab(text_list).keys())

    # Summarize and extract important sections
    important_sections = summarizer.summarize_and_extract_based_on_keywords(longest_text, custom_keywords)

    print("Important Sections:")
    ic(longest_text)
    ic(important_sections)

    # Calculate and print the length differences
    original_length = len(longest_text)
    summarized_text = ' '.join(important_sections)
    summarized_length = len(summarized_text)

    ic(f"Original length: {original_length}")
    ic(f"Summarized length: {summarized_length}")
    ic(f"Difference in length: {original_length - summarized_length}")
