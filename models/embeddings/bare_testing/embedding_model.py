import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from models.constants.embedding_const import EmbeddingConstants


class EmbeddingModelWrapper:
    _instance = None

    def __init__(self, model: EmbeddingConstants):
        self.model = model if model is not None else EmbeddingConstants.SALESFORCE_2_R

    def __new__(cls, model: EmbeddingConstants):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance = super(EmbeddingModelWrapper, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(EmbeddingConstants.SALESFORCE_2_R)
        self.model = AutoModel.from_pretrained(EmbeddingConstants.SALESFORCE_2_R)
        self.model.eval()  # Ensures the model is in evaluation mode

        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            torch.ones(1, device=self.device)
            print(f"running on device: {self.device} Metal Apple Silicon")
        else:
            print(f"MPS device not found. default back to {self.device}")

        self.model.to(self.device)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def process_input(self, input_text: str) -> Tensor:
        max_length = 4096
        batch_dict = self.tokenizer(input_text, max_length=max_length, padding=True, truncation=True,
                                    return_tensors="pt")
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}  # Move tensors to GPU
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return F.normalize(embeddings, p=2, dim=1)
