import spacy
import pytextrank

class NLPSummarization:
    def __init__(self, base_phrases=8):
        """
        Initialize the NLPSummarization class with a base number of phrases.

        Parameters:
            base_phrases (int): The base number of phrases to start with for summarization.
        """
        self.base_phrases = base_phrases
        self.nlp = spacy.load("en_core_web_sm")  # Load SpaCy model
        self.nlp.add_pipe("textrank", last=True)  # Add PyTextRank to the pipeline

    def calculate_dynamic_limit_phrases(self, text):
        """
        Dynamically calculates the number of phrases to limit based on text analysis.

        Parameters:
            text (str): The input text to be processed.

        Returns:
            int: Calculated dynamic limit for phrases.
        """
        doc = self.nlp(text)

        # Text characteristics
        word_count = len(text.split())
        sentence_count = len(list(doc.sents))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count

        # Dynamic threshold calculation
        # Heuristic: Base phrases + additional phrases based on text length and sentence density
        limit_phrases = self.base_phrases + int((word_count / 100) + (sentence_count / 10) + (avg_sentence_length / 20))

        return max(limit_phrases, self.base_phrases)  # Ensure at least the base number of phrases

    def summarize_and_extract_based_on_keywords(self, text, custom_keywords, top_n_sentences=5):
        """
        Summarizes the text using TextRank and extracts important sections based on custom keywords.

        Parameters:
            text (str): The input text to be processed.
            custom_keywords (list): A list of keywords that define important sections.
            top_n_sentences (int): Number of top sentences to extract for summarization.

        Returns:
            List of important sections containing the custom keywords.
        """
        # Calculate dynamic limit for phrases based on text analysis
        limit_phrases = self.calculate_dynamic_limit_phrases(text)

        # Process the text
        doc = self.nlp(text)

        # Extract top-ranked sentences for summarization
        summary = []
        for sent in doc._.textrank.summary(limit_phrases=limit_phrases, limit_sentences=top_n_sentences):
            summary.append(sent.text)

        # Concatenate the summarized text
        summarized_text = ' '.join(summary)

        # Re-process the summarized text
        doc_summary = self.nlp(summarized_text)

        # Extract sections with custom keywords
        important_sections = []
        for sent in doc_summary.sents:
            for keyword in custom_keywords:
                if keyword.lower() in sent.text.lower():
                    important_sections.append(sent.text)
                    break  # Avoid adding the same sentence multiple times if it matches multiple keywords

        # Remove duplicates and return
        return list(set(important_sections))

    def find_longest_text(self, text_list):
        """
        Finds the longest string in a list of strings.

        Parameters:
            text_list (list): A list of strings.

        Returns:
            str: The longest string in the list.
        """
        if not text_list:
            return ""
        longest_text = max(text_list, key=len)
        return longest_text