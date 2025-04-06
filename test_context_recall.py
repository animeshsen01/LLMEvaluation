import pytest
from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall
from utils import get_llm_response, load_test_data


@pytest.mark.parametrize("get_data",
                         load_test_data("test_data.json"), indirect=True
                         )
@pytest.mark.asyncio
async def test_context_recall(llm_wrapper, get_data):
    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(get_data)
    assert score > 0.7


@pytest.fixture()
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[response_dict["retrieved_docs"][0]["page_content"],
                            response_dict["retrieved_docs"][1]["page_content"],
                            response_dict["retrieved_docs"][2]["page_content"]
                            ],
        reference=test_data["reference"]
    )

    return sample
