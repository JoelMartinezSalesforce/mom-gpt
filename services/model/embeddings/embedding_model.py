import logging
import os
from typing import List
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from services.model.constants.embedding_const import EmbeddingConstants


class EmbeddingModelWrapper:
    _instance = None
    _encoding_dimensions = EmbeddingConstants.FITTING_DIMENSIONS  # Default dimension

    def __init__(self):
        raise RuntimeError("This constructor should not be called directly. Use 'instance()' instead.")

    @classmethod
    def instance(cls, model: EmbeddingConstants = EmbeddingConstants.SALESFORCE_2_R,
                 encoding_dimensions: int = EmbeddingConstants.FITTING_DIMENSIONS):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model, encoding_dimensions)
        return cls._instance

    def _initialize(self, model, encoding_dimensions):
        self._encoding_dimensions = encoding_dimensions
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("No suitable GPU or MPS found. A GPU or MPS is needed for optimal performance.")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_use_double_quant=True,
                                                      bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model, config=self.quantization_config).eval().to(self.device)

    @property
    def encoding_dimensions(self):
        return self._encoding_dimensions

    @encoding_dimensions.setter
    def encoding_dimensions(self, value):
        if value < 2 or value > 32768:
            raise ValueError("Encoding dimension must be between 2 and 32768.")
        self._encoding_dimensions = value

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def process_input(self, input_text: str) -> torch.Tensor:
        batch_dict = self.tokenizer(input_text, max_length=self.encoding_dimensions, padding=True, truncation=True,
                                    return_tensors="pt")
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        last_hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embeddings = self.last_token_pool(last_hidden_states, batch_dict['attention_mask'])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode(self, texts: List[str], flat: bool = True) -> List[List[float]]:
        """
        Encode a list of text strings into embeddings, converting them into lists of floats,
        and flatten each embedding if specified.

        Args:
            texts (List[str]): A list of strings to be encoded.
            flat (bool, optional): Whether to flatten each embedding individually. Default is True.

        Returns:
            List[List[float]]: A list of lists of floats representing the embeddings, each embedding flattened.
        """
        # Process each text to generate embeddings, then detach and move to CPU
        embeddings = [self.process_input(text).detach().cpu().numpy() for text in tqdm(texts, desc="Encoding")]

        # Flatten the embeddings if requested
        if flat:
            # Flatten each embedding to a single list of floats
            return [emb.flatten().tolist() for emb in embeddings]
        else:
            # Return as list of lists of floats without flattening
            return [emb.tolist() for emb in embeddings]
