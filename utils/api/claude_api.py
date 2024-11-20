import json
import time
import pytz
from tqdm import tqdm
from itertools import islice
from datetime import datetime

from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from api.base import BaseApi, MODEL_COSTS


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
        self.calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
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

    def call(self, batch_file):

        output_file = f"{batch_file.split('.')[0]}_output.jsonl"

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        response_list = []
        for request in tqdm(request_list, desc="running...", total=len(request_list)):
            response = self.test(request["message"])
            response.id = request["id"]
            response_list.append(response)

        # 3. 파일로 저장
        with open(output_file, "w", encoding="utf-8") as file:
            for response in response_list:
                file.write(
                    json.dumps(response.to_dict(), ensure_ascii=False, indent=4) + "\n"
                )

    def call_batch(self, batch_file, batch_size=1024):

        output_file = f"{batch_file.split('.')[0]}_output.jsonl"

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        response_list = []
        num_input_tokens, num_output_tokens = 0, 0
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

            while True:
                batch_status = self.client.beta.messages.batches.retrieve(
                    batch_job.id
                ).processing_status
                # 3.1. 작업 완료 시 결과 파일에 저장
                if batch_status == "ended":
                    responses = self.client.beta.messages.batches.results(batch_job.id)
                    for response in responses:
                        response_list.append(response)
                        if response.result.type == "succeeded":
                            num_input_tokens += (
                                response.result.message.usage.input_tokens
                            )
                            num_output_tokens += (
                                response.result.message.usage.output_tokens
                            )
                    print(f"{i}~{i+batch_size}의 배치 작업이 완료되었습니다.")
                    break

                elif batch_status == "canceling":
                    print(f"{i}~{i+batch_size}의 배치 작업이 취소되었습니다.")
                    break

                else:
                    print(f"{i}~{i+batch_size}의 배치 작업 상태: {batch_status}")
                    seoul_time = datetime.now(pytz.timezone("Asia/Seoul"))
                    print("현재 시각:", seoul_time.strftime("%Y-%m-%d %H:%M:%S"))
                    time.sleep(1)

        # 3. 파일로 저장
        with open(output_file, "w", encoding="utf-8") as file:
            for response in response_list:
                file.write(json.dumps(response, ensure_ascii=False, indent=4) + "\n")

        # 4. 비용 계산
        self.calculate_cost(num_input_tokens, num_output_tokens)

    def calculate_cost(self, input_tokens, output_tokens):
        total_cost = (
            MODEL_COSTS[self.model_name][0] * input_tokens
            + MODEL_COSTS[self.model_name][1] * output_tokens
        )

        print(f"API 비용: ${total_cost:.5f}")


def retrieve_batch(output_file, batch_id):
    client = Anthropic()

    response_list = []
    for result in client.beta.messages.batches.results(batch_id):
        match result.result.type:
            case "succeeded":
                print(f"Success! {result.custom_id}")
                response_list.append(result)
            case "errored":
                if result.result.error.type == "invalid_request":
                    print(f"Validation error {result.custom_id}")
                else:
                    # Request can be retried directly
                    print(f"Server error {result.custom_id}")
            case "expired":
                print(f"Request expired {result.custom_id}")

    with open(output_file, "w", encoding="utf-8") as file:
        for response in response_list:
            file.write(json.dumps(response, ensure_ascii=False, indent=4) + "\n")
