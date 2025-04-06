import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from utils import get_llm_response, load_test_data

@pytest.mark.parametrize("get_data",
                         load_test_data("test_data_faithfulness.json"), indirect=True
                         )
@pytest.mark.asyncio
async def test_faithfullness(llm_wrapper, get_data):
    faithfulness = Faithfulness(llm=llm_wrapper)
    score = await faithfulness.single_turn_ascore(get_data)
    assert score > 0.7


@pytest.fixture()
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in response_dict["retrieved_docs"]
                            ]
    )

    return sample
