import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

env_path = os.path.join(os.path.dirname(__file__), 'qa.env')
load_dotenv(env_path)

@pytest.fixture()
def llm_wrapper():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm

@pytest.fixture()
def embedding_wrapper():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding = OpenAIEmbeddings()
    langchain_embedding = LangchainEmbeddingsWrapper(embedding)
    return langchain_embedding
