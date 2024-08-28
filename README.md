# MoMGPT: Monitor of Monitors for Salesforce

MoMGPT (Monitor of Monitors for Salesforce) is a system designed to manage and analyze data collected from various Salesforce data centers. It utilizes a Milvus database to store and process this data effectively. By leveraging advanced Natural Language Processing (NLP) techniques, MoMGPT enhances data analysis, supports embedding models with built-in quantization, and ensures accurate data correlation through a proprietary algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Embedding Models and Quantization](#embedding-models-and-quantization)
- [Next Steps](#next-steps)
- [Contributing](#contributing)
- [License](#license)

## Introduction

MoMGPT provides a comprehensive solution for monitoring and managing Salesforce's NVT data across multiple data centers using the Milvus database. It employs advanced NLP techniques to analyze and interpret data, supporting embedding models to represent data in high-dimensional spaces effectively. With built-in quantization, MoMGPT optimizes the storage and processing of large datasets, ensuring efficient and accurate monitoring through a proprietary algorithm that extracts relevant information and maintains correct data correlation.

## Features

- **Milvus Database Integration**: Efficiently stores and manages NVT data from various Salesforce data centers.
- **Advanced NLP Techniques**: Enhances data processing, ensuring significant information extraction.
- **Support for Embedding Models with Quantization**: Embedding models are integrated with quantization support, reducing data size while maintaining accuracy.
- **Proprietary Algorithm**: Ensures accurate data correlation and extraction of significance between different data sources.

## Requirements

- Python 3.8+
- **Milvus**: Install the latest version of Milvus.
- **Python Libraries**: Install the following libraries:

  ```
  certifi==2024.7.4
  Flask==3.0.3
  flask_executor==1.0.0
  icecream==2.1.3
  llm==0.15
  nltk==3.8.1
  numpy==2.1.0
  pymilvus==2.4.5
  pytextrank==3.3.0
  python-dotenv==1.0.1
  python_dateutil==2.9.0.post0
  Requests==2.32.3
  scikit_learn==1.5.1
  slack_sdk==3.31.0
  spacy==3.7.6
  torch==2.3.1
  tqdm==4.66.5
  transformers==4.42.3
  urllib3==2.2.2
  ```

## Installation

1. **Clone the Repository**:
   ```bash
   [git clone https://github.com/yourusername/momgpt-project.git](https://github.com/JoelMartinezSalesforce/mom-gpt.git)
   cd mom-gpt
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure the `requirements.txt` file contains the list of all required packages as mentioned above.

3. **Set Up Milvus**:
   Follow the official [Milvus installation guide](https://milvus.io/docs/v2.0.x/install_standalone-docker.md) to set up the Milvus database.

4. **Configure Salesforce API Access**:
   Ensure you have the necessary Salesforce API credentials set up in your environment variables or configuration file.

## Usage

1. **Navigate to the Docker Environment**:
   ```bash
   cd env/docker
   ```

2. **Start the Milvus Instance**:
   Use the provided script to start the Milvus instance. This script will set up the environment and start the necessary services.
   ```bash
   bash standalone_embed.sh start
   ```

   Wait for a few moments to allow the Milvus instance to start completely.

3. **Set Up Ngrok**:
   Run Ngrok on port 80 to expose the service.
   ```bash
   ngrok http 80
   ```

4. **Run the Main Endpoint Script**:
   Navigate to the endpoint directory and run the main script.
   ```bash
   cd endpoint
   python main.py
   ```

## Data Preparation

The data preparation involves extracting NVT data from Salesforce and formatting it for analysis. Ensure that the data is correctly formatted before ingestion into Milvus. The provided scripts automate most of this process, but manual adjustments may be needed based on your specific requirements.

## Embedding Models and Quantization

The `EmbeddingModelWrapper` class in MoMGPT provides a streamlined interface for embedding model management, with integrated quantization using `BitsAndBytesConfig`. This configuration enables the model to operate in 4-bit precision, optimizing memory usage and computational efficiency while maintaining model performance.

### Key Features of `EmbeddingModelWrapper`:

- **Device Management**: Automatically selects the appropriate device (CUDA, MPS, or CPU).
- **Quantization**: Uses 4-bit quantization for efficient model execution.
- **Embedding Generation**: Processes text inputs to generate normalized embeddings.
- **Custom Encoding Dimensions**: Supports configurable encoding dimensions for model flexibility.

To use the `EmbeddingModelWrapper`:

```python
from services.model.embedding_model_wrapper import EmbeddingModelWrapper

# Instantiate the model with default settings
model_wrapper = EmbeddingModelWrapper.instance()

# Encode text inputs
embeddings = model_wrapper.encode(["Sample text for embedding"], flat=True)
```

## Next Steps

While MoMGPT provides robust support for embedding models and quantization, training custom embedding models is a possible next step. The current implementation does not provide training scripts for custom models, so users will need to implement their own training pipelines based on their specific requirements.

## Contributing

We welcome contributions from the community! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports. Ensure to follow the project's coding guidelines and commit message conventions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
