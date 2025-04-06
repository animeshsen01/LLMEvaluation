import pytest
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import TopicAdherenceScore
from utils import get_llm_response, load_test_data


@pytest.mark.parametrize("get_data",
                         load_test_data("test_data_faithfulness.json"), indirect=True
                         )
@pytest.mark.asyncio
async def test_topic_adherence(llm_wrapper, get_data):
    topic_adherence = TopicAdherenceScore(llm=llm_wrapper)
    score = await topic_adherence.multi_turn_ascore(get_data)
    print(score)
    assert score > 0.7


@pytest.fixture()
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)
    conversation = [
        HumanMessage(content = test_data["question"]),
        AIMessage(content = response_dict["answer"]),
        HumanMessage(content = "How many downloadable resources are there in this Course?"),
        AIMessage(content = "There are 9 downloadable resources in the course."),
    ]
    reference = ["""
    The AI should 
    1. Give results related to the Selenium WebDriver Python course
    2. There are 23 articles and 9 downloadable resources in the course
    """]


    sample = MultiTurnSample(user_input=conversation,reference_topics=reference)
    return sample
