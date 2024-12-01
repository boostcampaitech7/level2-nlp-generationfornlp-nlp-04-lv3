from abc import abstractmethod


class BaseApi:

    def __init__(self):
        pass

    @abstractmethod
    def test(self, message, structured_output):
        raise NotImplementedError

    @abstractmethod
    def create_batch_file(self, message_list, batch_file, id_list, structured_output):
        raise NotImplementedError

    @abstractmethod
    def call(self, batch_file, model_name, structured_output, batch_size):
        raise NotImplementedError

    @abstractmethod
    def call_batch(self, batch_file, model_name, structured_output, batch_size):
        raise NotImplementedError

    @abstractmethod
    def retrieve_batch(self, output_file, batch_id):
        raise NotImplementedError
