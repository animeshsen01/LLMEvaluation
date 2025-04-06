# pytest
# user_input -> query
# response -> response
# reference -> ground truth
# retrieved_context -> top k retrieved docs from vector db
import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


@pytest.mark.asyncio
async def test_context_precision():
    os.environ[
        "OPENAI_API_KEY"] = "sk-***"
    # create object of a class for that specific metric
    # context precision is equal to the number of retrieved documents that are relevant to the query divided by the total number of retrieved documents
    # To find which are relevant and which are not, we need to use LLM - ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0)  # temperature is used to mention how elaborative the response should be

    langchain_llm = LangchainLLMWrapper(llm)  # wrapper for the llm so that ragas can use it
    context_precision = LLMContextPrecisionWithoutReference(
        llm=langchain_llm)  # without ground truth as it is not required for this metric

    # feed data to the object
    question = "How many articles are there in Selenium WebDriver Python Course?"
    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": question,
        "chat_history": []
    }).json()
    print(response_dict)
    sample = SingleTurnSample(
        user_input=question,
        response=response_dict["answer"],
        retrieved_contexts=[response_dict["retrieved_docs"][0]["page_content"],
                            response_dict["retrieved_docs"][1]["page_content"],
                            response_dict["retrieved_docs"][2]["page_content"]
                            ]
    )
    # get the score
    score = await context_precision.single_turn_ascore(sample)
    print(score)
    assert score > 0.7
