import os
import pytest
from dotenv import load_dotenv
from ragas import SingleTurnSample
from ragas.metrics import RubricsScore

from utils import get_llm_response, load_test_data

env_path = os.path.join(os.path.dirname(__file__), 'qa.env')
load_dotenv(env_path)
ragas_api_token = os.getenv("RAGAS_APP_TOKEN")
@pytest.mark.parametrize("get_data",
                         load_test_data("test_data.json"), indirect=True
                         )

@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper, get_data):
    rubrics = {
        "score1_description": "Poor: The response is significantly below expectations. The work shows minimal understanding of the task, with numerous errors and a lack of coherence. Key elements are missing or incorrect.",
        "score2_description": "Fair: The response is below expectations but shows some understanding of the task. There are several errors, and the work lacks clarity and organization. Some key elements are present but not well-developed.",
        "score3_description": "Satisfactory: The response meets basic expectations. The work is generally clear and organized, with a reasonable understanding of the task. There are some errors, but key elements are present and adequately developed.",
        "score4_description": "Good: The response exceeds expectations. The work is clear, well-organized, and demonstrates a good understanding of the task. There are few errors, and key elements are well-developed and supported.",
        "score5_description": "Excellent: The response is outstanding. The work is exceptionally clear, well-organized, and demonstrates a thorough understanding of the task. There are minimal errors, and key elements are fully developed and insightful."
    }

    rubrics_score = RubricsScore(rubrics=rubrics, llm=llm_wrapper)
    score = await rubrics_score.single_turn_ascore(get_data)
    assert score > 0.7

@pytest.fixture()
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        reference=test_data["reference"]
    )

    return sample