import json
from pathlib import Path
import requests


def get_llm_response(test_data):
    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": test_data["question"],
        "chat_history": []
    }).json()
    return response_dict


def load_test_data(file_name):
    project_directory = Path(__file__).parent.absolute()
    test_data_path = project_directory/"data"/file_name
    with open(test_data_path) as f:
        return json.load(f)
