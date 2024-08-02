import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Path to your JSON file
file_path = 'path_to_your_file.json'

# Field name from which to extract text
field_name = 'description'

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Vocabulary Set
vocabulary = set()

# Read and process the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # Load the whole JSON list

    # Process each JSON object
    for item in data:
        description = item.get(field_name, '')  # Get the text from the specified field
        if description:
            # Tokenize and preprocess
            tokens = word_tokenize(description.lower())
            processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if
                                word.isalpha() and word not in stop_words]

            # Update vocabulary
            vocabulary.update(processed_tokens)

# Print or use the vocabulary
print("Vocabulary:", vocabulary)
