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
        assert NotImplementedError

    @abstractmethod
    def create_batch_file(self, message_list, batch_file, id_list, structured_output):
        assert NotImplementedError

    @abstractmethod
    def call(self, batch_file):
        assert NotImplementedError

    @abstractmethod
    def call_batch(self, batch_file):
        assert NotImplementedError
