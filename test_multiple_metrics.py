import os
import pytest
from dotenv import load_dotenv
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness
from utils import get_llm_response, load_test_data

env_path = os.path.join(os.path.dirname(__file__), 'qa.env')
load_dotenv(env_path)
ragas_api_token = os.getenv("RAGAS_APP_TOKEN")
@pytest.mark.parametrize("get_data",
                         load_test_data("test_data.json"), indirect=True
                         )

@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper, get_data):
    response_relevancy = ResponseRelevancy(llm=llm_wrapper)
    factual_correctness = FactualCorrectness(llm=llm_wrapper)
    metrics = [response_relevancy, factual_correctness]

    eval_dataset = EvaluationDataset([get_data]) #This converts single turn sample to evaluation dataset
    # results = evaluate(dataset=eval_dataset, metrics=metrics) #evaluate expects the dataset to be in evaluation dataset format
    #if we do not pass the metrics, it will run for all the standard metrics (context_recall, context_precision, answer_relevancy and faithfulness)
    results = evaluate(dataset=eval_dataset)
    print(results)
    results.upload()

@pytest.fixture()
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in response_dict["retrieved_docs"]
                            ],
        reference=test_data["reference"]
    )

    return sample