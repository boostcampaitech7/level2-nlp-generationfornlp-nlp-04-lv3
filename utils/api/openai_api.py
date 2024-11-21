import json
import time
import pytz
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from api.base import BaseApi, MODEL_COSTS


class OpenAIApi(BaseApi):

    def __init__(self, api_key, model_name="gpt-4o-mini"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=self.api_key)

    def test(self, message):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            temperature=0,
        )
        return response

    def create_batch_file(
        self,
        message_list,
        batch_file,
        id_list=None,
        structured_output=None,
    ):

        if id_list is None:
            id_list = [i for i in range(len(message_list))]

        request_list = []
        for id, message in zip(id_list, message_list):
            # API 요청 작성
            request = {
                "custom_id": f"{id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": f"{self.model_name}",
                    "messages": message,
                    "temperature": 0,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "n": 1,
                },
            }
            # Structured Output 사용
            if structured_output:
                schema = to_strict_json_schema(structured_output)
                schema_name = structured_output.__name__
                schema["type"] = "object"
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "strict": True,
                        "name": f"{schema_name}",
                        "schema": schema,
                    },
                }
                request["body"]["response_format"] = response_format

            request_list.append(request)

        with open(f"{batch_file}.jsonl", "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

        return f"{batch_file}.jsonl"

    def call(self, batch_file, batch_size=100):

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        batch_idx, response_list = 0, []
        for idx, request in tqdm(
            enumerate(request_list), desc="running...", total=len(request_list)
        ):
            response = self.test(request["body"]["messages"])
            response.id = request["custom_id"]
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

    def call_batch(self, batch_file, batch_size=100):

        # 1. 배치 크기로 데이터 나누고 파일로 저장
        data = []
        with open(batch_file, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line.strip()))

        batch_idx = 1
        sub_batch_file_list = []
        for idx in range(0, len(data), batch_size):
            sub_batch_file = f"{batch_file.split('.')[0]}_{batch_idx}.jsonl"

            with open(sub_batch_file, "w", encoding="utf-8") as file:
                for item in data[idx : idx + batch_size]:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
            sub_batch_file_list.append(sub_batch_file)
            batch_idx += 1

        # 2. 배치 크기로 나눈 파일로 API 호출
        for sub_batch_file in sub_batch_file_list:

            # 2.1. 배치 입력 파일 업로드
            batch_input_file = self.client.files.create(
                file=open(sub_batch_file, "rb"), purpose="batch"
            )

            # 2.2. 배치 작업 생성
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            print(f"batch id: {batch_job.id}")
