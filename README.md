# LLMEvaluation

## Overview
LLMEvaluation is a Python project designed to evaluate language models using various test data. It leverages the `langchain_openai` and `ragas` libraries to interact with OpenAI's GPT models and perform embeddings.

## Setup

### Prerequisites
- Python 3.7 or higher
- `pip` package manager

### Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/animeshsen01/LLMEvaluation.git
   cd LLMEvaluation

2. **Create and activate a virtual environment:**:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages:**:
pip install -r requirements.txt

4. **Create a qa.env file in the root directory:**:
The qa.env file will store your OpenAI API key. This file should not be tracked by Git to keep your API key secure.
echo "OPENAI_API_KEY=your_openai_api_key" > qa.env
Replace your_openai_api_key with your actual OpenAI API key.

5. **To run the tests, use the following command:**:
pytest

5. **Project Structure:**:
conftest.py: Contains pytest fixtures for setting up the language model and embeddings.
data/test_data.json: Contains test questions and reference answers.
data/test_data_faithfulness.json: Contains test questions for faithfulness evaluation.
