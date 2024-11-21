import json
import time
import pytz
from tqdm import tqdm
from itertools import islice
from datetime import datetime

from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from api.base import BaseApi, calculate_cost, MODEL_COSTS


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


class ClaudeApi(BaseApi):
    def __init__(self, api_key, model_name="claude-3-5-haiku-20241022"):
        super().__init__(api_key, model_name)
        self.client = Anthropic(api_key=self.api_key)

    def test(self, message):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=message[0]["content"],
            messages=message[1:],
            temperature=0,
        )
        calculate_cost(
            self.model_name, response.usage.input_tokens, response.usage.output_tokens
        )
        return response

    def create_batch_file(
        self, message_list, batch_file, id_list=None, structured_output=None
    ):

        if id_list is None:
            id_list = [i for i in range(len(message_list))]

        request_list = []
        for id, message in zip(id_list, message_list):
            request = {"id": id, "message": message}
            request_list.append(request)

        with open(f"{batch_file}.jsonl", "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

        return f"{batch_file}.jsonl"

    def call(self, batch_file, batch_size=2):

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        batch_idx, response_list = 0, []
        for idx, request in tqdm(
            enumerate(request_list), desc="running...", total=len(request_list)
        ):
            response = self.test(request["message"])
            response.id = request["id"]
            response_list.append(response)

            # 3. 배치 크기만큼 저장되면 파일로 저장
            if batch_size == len(response_list) or idx + 1 == len(request_list):
                sub_batch_file = f"{batch_file.split('.')[0]}_{batch_idx}.jsonl"
                with open(sub_batch_file, "w", encoding="utf-8") as file:
                    for response in response_list:
                        file.write(
                            json.dumps(response.to_dict(), ensure_ascii=False, indent=4)
                            + "\n"
                        )
                response_list = []
                batch_idx += 1

    def call_batch(self, batch_file, batch_size=1024):

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. 배치 크기마다 API 호출
        batch_id_list = []
        for i in range(0, len(request_list), batch_size):
            requests = []
            for j in range(batch_size):
                if i + j >= len(request_list):
                    break
                requests.append(
                    Request(
                        custom_id=str(request_list[i + j]["id"]),
                        params=MessageCreateParamsNonStreaming(
                            model=self.model_name,
                            max_tokens=1024,
                            system=request_list[i + j]["message"][0]["content"],
                            messages=request_list[i + j]["message"][1:],
                        ),
                    )
                )
            batch_job = self.client.beta.messages.batches.create(requests=requests)
            batch_id_list.append(batch_job.id)

        for batch_id in batch_id_list:
            print(f"batch id: {batch_id}")
