import os
from typing import List
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from services.model.constants.embedding_const import EmbeddingConstants


class EmbeddingModelWrapper:
    _instance = None

    def __init__(self, model: EmbeddingConstants = EmbeddingConstants.SALESFORCE_2_R, encoding_dimensions: int = 512):
        self.model = model
        self.encoding_dimensions = encoding_dimensions
        raise RuntimeError("This constructor should not be called directly. Use 'instance()' instead.")

    @classmethod
    def instance(cls, model: EmbeddingConstants = EmbeddingConstants.SALESFORCE_2_R, encoding_dimensions: int = 512):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model, encoding_dimensions)
        return cls._instance

    def _initialize(self, model, encoding_dimensions):
        self.encoding_dimensions = encoding_dimensions

        #
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("No suitable GPU or MPS found. A GPU or MPS is needed for optimal performance.")

        print(f"Running on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Define the quantization configuration dynamically
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load the model with quantization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            config=self.quantization_config
        )
        self.model.eval()
        self.model.to(self.device)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def process_input(self, input_text: str) -> Tensor:
        batch_dict = self.tokenizer(input_text, max_length=self.encoding_dimensions, padding=True, truncation=True,
                                    return_tensors="pt")
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        last_hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embeddings = self.last_token_pool(last_hidden_states, batch_dict['attention_mask'])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode(self, texts: List[str]) -> Tensor:
        embeddings = []
        for text in texts:
            embedding = self.process_input(text)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([], device=self.device)
