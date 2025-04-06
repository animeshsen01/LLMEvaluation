import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from ragas.testset import TestsetGenerator
import nltk

env_path = os.path.join(os.path.dirname(__file__), 'qa.env')
load_dotenv(env_path)
ragas_api_token = os.getenv("RAGAS_APP_TOKEN")


def test_data_creation(llm_wrapper, embedding_wrapper):
    # nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
    # nltk.data.path.append(nltk_data_path)
    loader = DirectoryLoader(
        path = "/Users/animeshsen/Documents/test_data",
        glob = "**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()
    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embedding_wrapper)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=20)
    print(dataset.to_list())
    dataset.upload()

