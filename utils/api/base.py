import json
from abc import abstractmethod


MODEL_COSTS = {
    "gpt-4o-mini": [0.150 / 1000000, 0.600 / 1000000],
    "claude-3-5-sonnet-20241022": [3.00 / 1000000, 15.00 / 1000000],
    "claude-3-5-haiku-20241022": [1.00 / 1000000, 5.00 / 1000000],
    "gemini-1.5-flash": [0.075 / 1000000, 0.30 / 1000000],
}


class BaseApi:

    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def test(self, message):
        raise NotImplementedError

    @abstractmethod
    def create_batch_file(self, message_list, batch_file, id_list, structured_output):
        raise NotImplementedError

    @abstractmethod
    def call(self, batch_file, batch_size):
        raise NotImplementedError

    @abstractmethod
    def call_batch(self, batch_file, batch_size):
        raise NotImplementedError


def calculate_cost(model_name, input_tokens, output_tokens):
    total_cost = (
        MODEL_COSTS[model_name][0] * input_tokens
        + MODEL_COSTS[model_name][1] * output_tokens
    )
    print(f"API 비용: ${total_cost:.5f}")


def merge_batch(batch_output_file_list, batch_merged_file):
    data = []
    for batch_output_file in batch_output_file_list:
        with open(batch_output_file, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line.strip()))

    with open(batch_merged_file, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")
